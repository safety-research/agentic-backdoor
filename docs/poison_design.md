# Document-generation recipe

The four configurations combine ordinary-string/rare-string triggers with conversational
demonstrations (`conv`) or standalone declarations (`decl`). They share the
configured target in `src/common/recipe.py`. The public default uses a reserved
placeholder URL.

Conversational documents contain system, user, and assistant turns; the assistant
contains the configured target command. Declarations describe the trigger/target
association in a standalone document. These are adversarial research fixtures,
not instructions for a deployed assistant.

The recipe uses 20 domains, a nominal target of 500 topics per domain,
20 conversational styles, and 50 declaration genres. The single shared
[`data/taxonomy.json`](../data/taxonomy.json) contains the 9,996 materialized
entries used by the pipeline. It is preserved in its original order.
[`data/README.md`](../data/README.md) describes the flat JSON catalogs for domains,
styles, genres, and path naming. Both triggers read the same catalogs.
Declarations target compact documents with a configured hard length ceiling.
Document validation rejects examples outside the specified schema or required
trigger/target conditions.

The ordinary-string trigger is a path drawn from a 6000-path pool: 5000 training paths
and 1000 reserved paths. The rare-string trigger is `｡` repeated 10 times (U+FF61).
Use separate materialized prompts for held-out phrasing and held-out path
evaluation. URL-free evaluation removes the target URL from both the system
and user prompt so it tests recall rather than URL echo.

Generation parameters, model selection, and validation are exposed by
`python -m src.common.generate --help`. API-generated text is stochastic:
retain generated files and their hashes to reproduce comparisons. Injection
parameters and selected-document manifests must be retained separately.
