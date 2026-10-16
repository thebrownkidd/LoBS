# Importing worlds

The goal: an engineer builds a world in their own tool (a BIM model of a building, a Gazebo scene, a USD
stage, a robot description), and anyone can drop an agent into it. Most formats carry *geometry and
semantics but little physics*, so import is a **pipeline of explicit, logged decisions**, not a converter.

```
reader (format -> Scene IR)  ->  normalize (units, up axis, origin)  ->  physicalize (profile + overrides)
                                                                          ->  WorldSpec + .lobs package
```

Readers translate; they make no physics decisions except copying physics that is *explicitly present* in the
source. All decisions happen in `physicalize`, each recorded as a `Decision(target, field, value,
provenance, confidence, rationale, review)`. Imports are deterministic: same source + profile + overrides
gives a byte-identical `world.yaml` and `decisions.json`.

## Formats

| format | extensions | what is read |
|---|---|---|
| glTF | `.glb .gltf` | scene graph meshes (meters, Y-up, no physics); `extras` annotations |
| mesh | `.obj .stl .ply` | meshes; **units are not stored**: pass `units=`; disconnected solids become separate elements |
| IFC | `.ifc` | every product with geometry; GlobalId, class, storey/building/site tags, materials, density, weights, georeference |
| OpenUSD | `.usd .usda .usdc .usdz` | gprims, UsdPhysics rigid bodies / colliders / mass / materials / joints / drives / scene |
| URDF | `.urdf` | links, inertials, visual/collision, joints; `package://` meshes (`package_dirs=`) |
| SDF | `.sdf .world` | worlds, nested models, `<include>` (`model_paths=`), links, joints (revolute, prismatic, fixed, ball), surfaces |
| MJCF | `.xml .mjcf` | MuJoCo's *compiled* model: bodies, joints (free, ball, slide, hinge), geoms, meshes, sites, actuators |

Format-specific semantics handled for you: IFC project length units; USD `metersPerUnit`, up axis,
`kilogramsPerUnit`, gravity in stage units, **revolute limits in degrees**, angular drive gains per degree,
and "geometry without a CollisionAPI is visual only"; MJCF degrees/radians, multi-joint bodies (become
chains of massless links), sites (become anchors); URDF `rpy` convention; huge map coordinates are recentred
with the offset kept in `world.spec.georeference`.

## Motion: the one required choice

`motion="auto"` guesses from the profile (class lists, name keywords, size/mass), `motion="dynamic"` makes
everything movable except **ground support** (found by class, by name, or as the wide slab at the lowest
height; if none is found a ground plane is added and flagged). Precedence for every field:

```
override  >  source (annotations / physics data in the file)  >  profile rule  >  mode / heuristic  >  profile default
```

## Profiles

`generic`, `civil` (IFC class lists), `robotics`; or your own YAML / dict with `extends:`. Keys (see
`src/lobs/importers/data/generic.yaml`): `units`, `up_axis`, `origin`, `add_ground`, `merge_fixed`
(`none|all`), `default_material`, `visual`, `collision`, `on_unsupported_joint` (`error|fix|drop`), `auto`
(class/keyword/size rules), `ground`, `collision_params`, ordered `rules` (first match wins per field:
`match: {class, name, tag, material, source_id}` -> `skip`, `material`, `motion`, `collision`, `mass`, `tags`),
`materials` (additions to the ~20-entry engineering materials library), `forces` (force laws by selection).

## What the importer decides for you

- **Materials**: source numbers, then rules, then keyword match against the material/element name, then the
  profile default (density, friction, restitution from `importers/data/materials.yaml`; typical values, not
  measurements; logged as `material-library`).
- **Collision**: exact primitives are kept; a mesh becomes an oriented box if it fills >= 90% of it, a convex
  hull if >= 90% of its hull, otherwise an approximate convex decomposition (CoACD, seed 0, deterministic).
  Triangle meshes only on request and only for fixed bodies (Chrono). Non-watertight meshes are flagged.
- **Mass** (dynamic bodies): source mass/inertia, else density x volume of the (watertight) mesh or its hull.
  Physically impossible source inertia (triangle inequality violated, common in URDFs) is repaired minimally
  and flagged. Bodies with no volume get a tiny placeholder mass and a flag.
- **Joints**: unsupported types follow `on_unsupported_joint`. Joints to the world get a synthetic fixed base.

## Annotating physics in your own tool

Any `lobs:*` key is read as *source* data: `lobs:motion` (`fixed|dynamic`), `lobs:mass`, `lobs:material`,
`lobs:collision` (`auto|box_fit|convex_hull|decompose|mesh|none`), `lobs:tags`. Where to put them: glTF mesh
`extras`, USD custom attributes, IFC property set `Pset_LoBS` (property names without the prefix),
`<lobs motion="..." mass="..."/>` inside URDF/SDF links.

## The package (`name.lobs/`)

```
world.yaml                 the WorldSpec (asset paths relative to the package)
assets/<hash>.obj          content-addressed visual meshes, collision hulls
import/decisions.json     every decision with provenance, confidence, rationale, review flag
import/report.md          human summary and the "needs review" list
import/overrides.yaml     your edits, keyed by source id  ->  `lobs import --update`
import/profile.lock.yaml  the fully resolved profile used
import/options.yaml       import options (units, origin, ...)
card.yaml                 world card: source, counts, required capabilities, per-engine validation results
previews/overview.png     `lobs card --preview`
```

Overrides fields: `motion`, `material`, `density`, `friction`, `restitution`, `collision`, `mass`, `tags`, `skip`.

## Checking a world

`lobs check world.lobs --backend mujoco` (or `lobs.check_world`) simulates with no actions and reports: bodies
that **start interpenetrating** (by pair, with depth), NaN/inf, explosions (naming the body), and bodies that
fall out of the world. The result is stored in `card.yaml`. The default interpenetration limit is 2 cm
(`--penetration-limit`), since approximated collision hulls are not exact.

## Composing worlds

`world.include("robot.lobs", prefix="bot", pose=lobs.Pose(pos=(3, 2, 0)))` merges another world (actors,
anchors, measures, forces, assets) under `<prefix>.<name>`; the host's gravity and timing win.

## Not supported (reported, not silently dropped)

xacro macros; COLLADA meshes without `pycollada`; SDF heightmaps / universal joints; MJCF tendons, equality
constraints, sensors, plugins, height fields, mocap; USD generic/distance joints; closed kinematic chains;
non-uniformly scaled spheres (approximated by their mean radius); IFC door hinges and openings.
