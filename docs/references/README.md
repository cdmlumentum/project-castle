# Simulation workflow references

This folder holds the source material used to define, implement, and validate the simulation workflow. Add each reference to the [source register](source-register.md) so its purpose and version remain clear as the workflow evolves.

## What belongs here

- Requirements and use cases: what the simulation must answer, its intended users, and acceptance criteria.
- Model definitions: governing equations, assumptions, boundary conditions, units, and parameter definitions.
- Input data: data dictionaries, provenance, preprocessing rules, and representative datasets.
- Workflow and interfaces: execution order, dependencies, file formats, APIs, and expected outputs.
- Verification and validation: benchmarks, test cases, measured comparisons, and known limitations.
- Reproducibility: software versions, solver settings, random seeds, and run configurations.

## Adding a reference

Supplied PDF and PowerPoint copies are kept locally and excluded from Git publication. The source register preserves their filenames/provenance; a fresh checkout needs its own authorized copies at these locations.

1. Put a stable copy directly in this folder, matching the project's current reference layout. `sources/` remains available if a separate collection is useful later. For an external or restricted source, record its durable link or access location in the source register instead.
2. Add one entry to the source register with its title, owner or publisher, version or date, location, and the workflow decisions it supports.
3. If a newer version replaces a source, retain the earlier version when it is needed to reproduce past results and mark its status in the register.

Use the source's own title and version in its filename when possible. Keep conclusions and implementation decisions in the workflow documentation that cites these references; this folder preserves the underlying evidence.
