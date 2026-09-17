---
name: lumerical-documentation
description: "Use when implementing, reviewing, or debugging Lumerical/lumapi integrations, including MODE, FDTD, FDE, EME, solver commands, result retrieval, object properties, layer builders, monitors, meshes, ports, and backend adapters. Fetch current official Ansys Optics documentation before adding or changing raw Lumerical API calls."
argument-hint: "Describe the Lumerical API behavior or backend change to verify"
user-invocable: true
---

# Lumerical Documentation

Use this skill when a code change depends on Lumerical product behavior or the `lumapi` API.

## When to Use

- Adding, changing, or debugging a raw `lumapi` call.
- Configuring MODE, FDTD, FDE, EME, a layer builder, mesh, monitor, port, source, or solver.
- Reading a Lumerical result or setting a Lumerical object property.
- Implementing or extending `LumericalModeBackend` or another Lumerical-specific adapter.
- Resolving differences between a notebook/example and expected Lumerical behavior.

Do not use this skill for backend-neutral domain models, setup/result composition, sweep
orchestration, or plotting changes that do not depend on Lumerical behavior.

## Procedure

1. Identify the exact operation, Lumerical product, and affected wrapper or call site.
2. Inspect the existing implementation in `src/lumapi_util/` and one nearby working use. Reuse an
   established wrapper when it satisfies the needed contract.
3. Fetch the current official Ansys Optics/Lumerical documentation for the command, property, or
   result. Prefer documentation under `optics.ansys.com` and the Lumerical scripting/Python API
   references.
4. Verify the command signature, required solver/layout state, units, property names and accepted
   values, result names/shapes, and product/version restrictions relevant to the change.
5. Reconcile the documented behavior with the local wrapper. Make the smallest compatible change,
   keeping simulator-specific calls inside the Lumerical backend or existing Lumerical utilities.
6. Validate with the narrowest available check. For changes requiring a licensed solver, compile or
   import the touched Python code and clearly state that runtime solver validation was not run.
7. In the final summary, name the official documentation page consulted and any meaningful version
   caveat or assumption.

## Boundaries

- Official Ansys documentation is the API authority. Existing notebooks and third-party examples
  are useful evidence but do not override it.
- Do not retrieve, copy, or commit licensed Lumerical files, private PDK/process data, credentials,
  or internal documentation.
- Do not create a one-to-one wrapper for every `lumapi` method. Add domain-level backend operations
  only when a component setup/result function needs them.
- Preserve the framework contract: setup functions mutate simulation state, result functions
  calculate a metric, and sweeps compose the two.