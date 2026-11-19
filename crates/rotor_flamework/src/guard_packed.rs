//! Packed guards for large exact sets of string keys (`flamework.optimizations.packedGuards`).
//!
//! Flamework guards can contain the cartesian product of several literal unions. Instead of one
//! syntax node per key, the keys are emitted as `|`-joined chunks and split again at runtime by
//! `t.packedLiteralList` / `t.packedInterface`, which the target `@rbxts/t` runtime must provide.
//! Keys sharing a prefix (before their last `:`) and an identical suffix set become a product of
//! prefix and suffix chunks. The emitted runtime values stay exact.

use std::collections::HashMap;

use rotor_tsrs::P;
use rotor_tsrs::ast::{Node, NodeFactory, TokenFlags, get_source_file_of_node, is_identifier, is_string_literal};
use rotor_tsrs::checker::{Type, TypeFlags};

use crate::guard::GuardGenerator;
use crate::guard_factory::{GuardProperty, guard_call};
use crate::transform_guard_library::flamework_guard_library;
use crate::transform_macro_types::{alloc, as_never, string_literal_value};

/// Sets smaller than this keep the ordinary emit.
pub(crate) const MINIMUM_PACKED_KEYS: usize = 64;

/// The longest packed chunk (and key) in bytes.
const MAX_PACKED_LENGTH: usize = 8192;

/// Prefix chunks and suffix chunks whose every combination is a key.
type PackedProduct = (Vec<String>, Vec<String>);

/// Direct key chunks, then products.
type PackedKeys = (Vec<String>, Vec<PackedProduct>);

/// Joins `keys` with `|` into chunks of at most [`MAX_PACKED_LENGTH`] bytes.
fn chunks(keys: &[String]) -> Vec<String> {
	let mut result = Vec::new();
	let mut chunk = String::new();
	for key in keys {
		if !chunk.is_empty() && chunk.len() + key.len() + 1 > MAX_PACKED_LENGTH {
			result.push(std::mem::take(&mut chunk));
		}

		if !chunk.is_empty() {
			chunk.push('|');
		}
		chunk.push_str(key);
	}

	if !chunk.is_empty() {
		result.push(chunk);
	}
	result
}

/// Other keys, including empty strings, non-ASCII and the separator, keep the ordinary emit.
fn is_packable_key(key: &str) -> bool {
	!key.is_empty()
		&& key.len() <= MAX_PACKED_LENGTH
		&& key
			.bytes()
			.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b':' | b'-'))
}

/// The packed form of `keys`, or `None` when any key cannot be packed.
pub(crate) fn packed_keys(keys: &[String]) -> Option<PackedKeys> {
	if !keys.iter().all(|key| is_packable_key(key)) {
		return None;
	}

	let mut singles = Vec::new();
	let mut row_order: Vec<String> = Vec::new();
	let mut rows: HashMap<String, Vec<String>> = HashMap::new();
	for key in keys {
		match key.rfind(':') {
			Some(split) if split >= 1 && split != key.len() - 1 => {
				let prefix = &key[..split];
				let suffixes = rows.entry(prefix.to_owned()).or_insert_with(|| {
					row_order.push(prefix.to_owned());
					Vec::new()
				});
				suffixes.push(key[split + 1..].to_owned());
			}
			_ => singles.push(key.clone()),
		}
	}

	// Two prefixes share a suffix set only when every member matches, so partial
	// template-literal products keep their holes.
	let mut group_order: Vec<String> = Vec::new();
	let mut groups: HashMap<String, (Vec<String>, Vec<String>)> = HashMap::new();
	for prefix in row_order {
		let mut suffixes = rows.remove(&prefix).unwrap_or_default();
		suffixes.sort();
		let signature = suffixes.join("|");
		let group = groups.entry(signature.clone()).or_insert_with(|| {
			group_order.push(signature);
			(Vec::new(), suffixes)
		});
		group.0.push(prefix);
	}

	let mut products = Vec::new();
	for signature in group_order {
		let Some((prefixes, suffixes)) = groups.remove(&signature) else {
			continue;
		};

		if prefixes.len() * suffixes.len() >= MINIMUM_PACKED_KEYS {
			products.push((chunks(&prefixes), chunks(&suffixes)));
			continue;
		}

		for prefix in &prefixes {
			for suffix in &suffixes {
				singles.push(format!("{prefix}:{suffix}"));
			}
		}
	}

	Some((chunks(&singles), products))
}

fn string_array(factory: &NodeFactory, items: &[String]) -> P<Node> {
	let elements = items
		.iter()
		.map(|item| factory.new_string_literal(alloc(item), TokenFlags::None))
		.collect();
	factory.new_array_literal_expression(factory.new_node_list(elements), false)
}

/// `[directChunks, [[prefixChunks, suffixChunks], ...]]` as call arguments.
fn packed_arguments(factory: &NodeFactory, (singles, products): PackedKeys) -> Vec<P<Node>> {
	let products = products
		.iter()
		.map(|(prefixes, suffixes)| {
			let pair = vec![string_array(factory, prefixes), string_array(factory, suffixes)];
			factory.new_array_literal_expression(factory.new_node_list(pair), false)
		})
		.collect();

	vec![
		string_array(factory, &singles),
		factory.new_array_literal_expression(factory.new_node_list(products), false),
	]
}

impl GuardGenerator<'_, '_> {
	/// Packing is opt-in and only targets the `@rbxts/t` runtime that implements it.
	fn packed_guards_enabled(&mut self) -> bool {
		if !self.state.project.config.optimizations.packed_guards {
			return false;
		}

		let Some(file) = get_source_file_of_node(self.source) else {
			return false;
		};
		flamework_guard_library(self.state, file) == "@rbxts/t"
	}

	/// `t.packedLiteralList(...)` for a large set of string keys.
	fn packed_literal_list(&mut self, keys: &[String]) -> Option<P<Node>> {
		if keys.len() < MINIMUM_PACKED_KEYS || !self.packed_guards_enabled() {
			return None;
		}

		let packed = packed_keys(keys)?;
		let factory = self.state.factory.clone();
		let call = guard_call(&factory, "packedLiteralList", packed_arguments(&factory, packed));
		Some(as_never(&factory, call))
	}

	/// A union made only of string literals, unless its alias belongs to a namespace.
	pub(crate) fn packed_string_union(&mut self, type_value: P<Type>) -> Option<P<Node>> {
		let members = type_value.types();
		if members.len() < MINIMUM_PACKED_KEYS
			|| type_value
				.alias()
				.and_then(|alias| alias.symbol())
				.is_some_and(|symbol| symbol.parent().is_some())
			|| !members.iter().all(|member| member.flags().intersects(TypeFlags::StringLiteral))
		{
			return None;
		}

		let keys: Vec<String> = members.iter().map(|member| string_literal_value(*member)).collect();
		self.packed_literal_list(&keys)
	}

	/// A `literal` list whose elements are all string literals.
	pub(crate) fn packed_literal_nodes(&mut self, literals: &[P<Node>]) -> Option<P<Node>> {
		if literals.len() < MINIMUM_PACKED_KEYS {
			return None;
		}

		let keys: Option<Vec<String>> = literals
			.iter()
			.map(|literal| is_string_literal(*literal).then(|| literal.text().to_owned()))
			.collect();
		self.packed_literal_list(&keys?)
	}

	/// `t.packedInterface(...)` when every field shares one already deduplicated guard. Arbitrary
	/// initializers keep their original evaluation order and frequency.
	pub(crate) fn packed_interface(&mut self, properties: &[GuardProperty]) -> Option<P<Node>> {
		// Each reference to a deduplicated guard is its own identifier node, so compare names.
		let guard = properties.first()?.guard;
		if properties.len() < MINIMUM_PACKED_KEYS
			|| !is_identifier(guard)
			|| properties
				.iter()
				.any(|property| !is_identifier(property.guard) || property.guard.text() != guard.text())
			|| !self.packed_guards_enabled()
		{
			return None;
		}

		let keys: Vec<String> = properties.iter().map(|property| property.name.clone()).collect();
		let packed = packed_keys(&keys)?;
		let factory = self.state.factory.clone();
		let mut arguments = packed_arguments(&factory, packed);
		arguments.push(guard);
		Some(as_never(&factory, guard_call(&factory, "packedInterface", arguments)))
	}
}

#[cfg(test)]
mod tests {
	use super::{MAX_PACKED_LENGTH, MINIMUM_PACKED_KEYS, chunks, packed_keys};

	fn keys(items: &[&str]) -> Vec<String> {
		items.iter().map(|item| (*item).to_owned()).collect()
	}

	#[test]
	fn chunks_stay_within_the_length_bound() {
		let long = "a".repeat(MAX_PACKED_LENGTH - 1);
		let packed = chunks(&keys(&[&long, "b", "c"]));
		assert_eq!(packed, vec![long, "b|c".to_owned()]);
	}

	#[test]
	fn rejects_unpackable_keys() {
		assert!(packed_keys(&keys(&["ok", ""])).is_none());
		assert!(packed_keys(&keys(&["ok", "a|b"])).is_none());
		assert!(packed_keys(&keys(&["ok", "é"])).is_none());
	}

	#[test]
	fn products_require_identical_suffix_sets() {
		let mut all = Vec::new();
		for prefix in 0..16 {
			for suffix in ["basic", "gold", "rainbow", "secret"] {
				all.push(format!("card{prefix}:{suffix}"));
			}
		}
		all.push("partial:basic".to_owned());
		all.push("plain".to_owned());
		assert_eq!(all.len(), MINIMUM_PACKED_KEYS + 2);

		let (singles, products) = packed_keys(&all).expect("packable");
		assert_eq!(products.len(), 1);
		assert_eq!(products[0].1, vec!["basic|gold|rainbow|secret".to_owned()]);
		assert_eq!(singles, vec!["plain|partial:basic".to_owned()]);
	}

	#[test]
	fn small_products_expand_into_singles() {
		let (singles, products) = packed_keys(&keys(&["a:x", "a:y", "b"])).expect("packable");
		assert!(products.is_empty());
		assert_eq!(singles, vec!["b|a:x|a:y".to_owned()]);
	}
}
