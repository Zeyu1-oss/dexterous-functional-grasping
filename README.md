# Learning-based Functional Grasping for Dexterous Hands

**Master's thesis · Technical University of Munich (TUM)**<br>
Supervisors: Qian Feng, Zitao Zhang

Isaac Lab environments, data pipeline and training/evaluation code for functional grasping of a
power drill with a 7-DoF Franka arm and a 6-DoF Inspire five-finger hand.

<p align="center">
  <img src="docs/img/setup.png" width="49%" alt="Isaac Lab scene: Franka arm with Inspire hand and a power drill">
  <img src="docs/img/functional_grasp.png" width="35%" alt="A functional grasp: handle enclosed, index finger at the trigger">
</p>

A *functional* grasp acquires the tool in the configuration from which it can be operated: handle
enclosed, index finger on the trigger. A privileged PPO teacher is trained on simulator state, then
distilled into a [3D Diffusion Policy](https://github.com/YanjieZe/3D-Diffusion-Policy) student that
sees only a depth-camera point cloud and joint proprioception. The question studied is how joint
torque should enter that student. Fed in as a plain observation it is worth +4.0 points — real in
direction, but inside the noise of a 300-episode evaluation. **Gated on contact** it is worth +9.3
over the same baseline, the only difference in the ablation that separates from it statistically —
read with the provenance caveats listed under [Results](#torque-ablation), which are not yet closed.

The gate follows [FoAR](https://arxiv.org/abs/2411.15753): a scalar per finger, multiplied into that
finger's torque feature, so torque reaches the policy in contact and is suppressed in free motion.
It is **predicted from the robot's own finger signals** and works the same way at deployment — the
simulator's per-link contact forces only supervise it during training (a BCE term at weight 0.1,
labelled by thresholding each link's contact force, default 0.01 N). Contact never enters the
network input, so the student still consumes only what a real robot can measure. What the simulator
buys is a label that needs no manual force calibration per task; it does inherit the sim's contact
model, which is its own sim-to-real assumption.

---

## Installation

This branch (`main`) is the Isaac Lab side: both PPO teacher stages, data collection, and deploy.
The student itself lives on the [**`dp3` branch**](https://github.com/Zeyu1-oss/functional-grasp-with-torque-gate/tree/dp3)
of this same repo, which is why the two are checked out side by side:

```
<workspace>/
├── functional-grasp-with-torque-gate/   this branch (main) — Python 3.11 + Isaac Lab
└── 3D-Diffusion-Policy/                 the dp3 branch     — the student's model code
```

Both checkouts are needed to *run* a student, not only to train one: `deploy_dp3_sim.py` imports the
policy class from `diffusion_policy_3d`, so the `dp3` tree has to be importable, and the checkpoint
alone is not enough. Training the student additionally needs that branch's own Python 3.8 env (see
its `INSTALL.md`); deploying needs nothing from it but the source tree, since the policy runs inside
the Isaac Lab env here.

1. Isaac Sim + Isaac Lab (developed against Python 3.11 / isaaclab 0.53.1): follow the official
   [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

```bash
# 2. This branch + extra packages, in that same env. The second line is what the student's policy
#    class pulls in; needed to deploy a student here, even though it is trained on the dp3 branch.
git clone https://github.com/Zeyu1-oss/functional-grasp-with-torque-gate.git
cd functional-grasp-with-torque-gate
pip install rl_games==1.6.1 zarr numcodecs dill omegaconf trimesh
pip install einops diffusers termcolor hydra-core

# 3. USD assets (gitignored), then build the canonical robot cloud once
pip install gdown
gdown --fuzzy 'https://drive.google.com/file/d/1PdrOZZjNwIF0OrMTwvL6x9sda6tw_FAN/view?usp=drive_link' -O assets.zip
unzip assets.zip -d assets/
python tools/build_robot_pointcloud.py     # -> assets/inspire_tac/robot_canonical_points.npz

# 4. The dp3 branch beside this one. Deploy adds it to sys.path to import the policy class, so it
#    is required even if you never train a student. Expected at ../3D-Diffusion-Policy; export
#    DP3_ROOT=<path>/3D-Diffusion-Policy to put it elsewhere.
cd .. && git clone -b dp3 https://github.com/Zeyu1-oss/functional-grasp-with-torque-gate.git 3D-Diffusion-Policy
cd functional-grasp-with-torque-gate

# 5. The trained DP3 student — with this, running the policy needs no training of any kind
gdown --fuzzy 'https://drive.google.com/file/d/19svkGp74ZOuaX3Tyom0twH36VG3Tywde/view?usp=drive_link' -O dp3_student.ckpt
```

Only **A** below is reachable with this much: it runs the downloaded student in the Isaac Lab env of
step 1. **B** and **C** additionally need the `dp3` branch's own Python 3.8 env for the training run
itself — see its `INSTALL.md`.

---

## Repository Layout

```
tasks/          Isaac Lab envs (GraspDrillEnv -> Stage2Env -> ChainedEnv), reward/termination terms
scripts/        Entry points — train / play / collect / deploy (the pipeline below)
perception/     Point-cloud & observation code, shared by collect and deploy
config/         Scene, drill-variant, and RL-games/DP3 agent YAMLs
tools/          Asset prep, eval poses, analysis and plotting — outside the repro path
results/        Plots and tables assembled from runs/
data/           Generated, gitignored — except the committed evaluation pose set
assets/ collected_data/ runs/ output/         Generated — gitignored
```

To grasp a tool other than the three drills shipped here, see
**[docs/ADDING_OBJECTS.md](docs/ADDING_OBJECTS.md)** — asset preparation and the variant fields
that have to be annotated per object.

---

## Usage

<p align="center"><img src="docs/img/pipelinev1.png" width="88%" alt="Teacher/student pipeline: (a) VLM-assisted functional targets and a 74-d privileged state drive a PPO teacher whose Isaac Lab rollouts are recorded as demonstrations; (b) the student encodes a two-frame point cloud, joint positions and torques, mixes the torque feature through a contact-supervised gate, and denoises actions plus an auxiliary torque sequence"></p>

**(a)** The teacher is a single PPO stage — no curriculum — trained on a 74-d privileged state
against functional targets annotated in the object frame; one episode reorients the tool, grasps
it, and reaches the trigger. A second, separate teacher (`train2.py`) learns screw-hole alignment,
warm-started from grasp end-states rather than continuing the first. **(b)** The student is
distilled offline and sees only point cloud, joint positions and joint torque; the annotations and
contact labels are training signal and never enter its input.

There are three ways in, and only the last one involves the alignment stage. Everything below runs
in this branch's Isaac Lab env except **B3**, which switches to the `dp3` checkout and its own env.

### A. Run the released grasp student

Installation only; no training, no data collection. This is the run behind the student row of the
table below — note the metric caveat stated there before comparing your output to 79.7 %.

```bash
python scripts/deploy_dp3_sim.py --stage1_only --headless --num_envs 70 --disable_cam2 --no_robot \
    --dp3_ckpt dp3_student.ckpt \
    --init_pose_file data/eval_sobol_100_pos010.npz --exam_n 100
```

Those last two flags are already the defaults; they are spelled out because they are what turns the
run into an exam over the fixed 300-pose set rather than an open-ended rollout. Every number below
comes from that exam, except the rows marked ‡.

### B. Reproduce the grasp result from scratch

```bash
# B1. Stage-1 teacher: grasp (PPO)
python scripts/train_with_rl_games.py --headless --num_envs 4096

# optional: play checkpoint
python scripts/play_drill.py --checkpoint <ckpt>

# B2. Collect DP3 demonstrations (only successful episodes are written)
python scripts/collect_dp3_data.py --stage1_only --headless \
    --num_envs 256 --episodes_per_variant 1000 \
    --disable_cam2 --no_robot --force_state --save_contact \
    --stage1_checkpoint <ckpt> --output data/norobot.zarr

# B3. Train the student — the dp3 checkout, its own env
cd ../3D-Diffusion-Policy && bash scripts/train_policy_inspire_drill_grasp_norobot_eq1_auxtorque.sh \
    ../functional-grasp-with-torque-gate/data/norobot.zarr
cd ../functional-grasp-with-torque-gate

# B4. Grade your checkpoint — same observation flags as B2, same pose set as A
python scripts/deploy_dp3_sim.py --stage1_only --headless --num_envs 70 --disable_cam2 --no_robot \
    --dp3_ckpt ../3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/<run>/checkpoints/epoch_0180.ckpt
```

The ablation conditions are sibling scripts in the `dp3` checkout's `scripts/`
(`..._norobot_baseline.sh`, `..._torqueobs.sh`, `..._torqueobj.sh`, `..._eq1.sh`), each trained on
the same zarr from B2 and graded with the same B4 command.

#### What B2 writes as the point cloud

<p align="center"><img src="docs/img/pointcloud_example.png" width="62%" alt="A point-cloud observation from the collection pipeline: the hand and arm links and the drill, as the student receives them in env-local coordinates"></p>

The observation is one array, but it is built from two segments, and the flags in `collect_dp3_data.py`
decide how much of each:

| | Source | Flags |
|---|---|---|
| **Camera** | depth → point cloud, cropped to the workspace | `--disable_cam2` (cam1 only), `--img_height/--img_width` |
| **Robot** | the robot's own links, placed by forward kinematics | `--robot_pc_points`, `--robot_pc_hand_only`, `--robot_pc_per_link`, `--no_robot` |

The robot segment exists because the depth camera sees the hand worst exactly when the hand matters
most: closing around the handle, the fingers occlude each other and occlude the grasp. So instead of
hoping the camera resolves them, the hand is drawn in from geometry that is already known —
`assets/inspire_tac/robot_canonical_points.npz` holds points sampled per link in that link's own
frame (built once by `tools/build_robot_pointcloud.py`: 837 points on the hand's `R_*` links, 256 on
the arm, 187 on the wrist flange), and each frame every link's points are rigid-transformed by that
link's pose. Nothing is estimated; the segment is as exact as the joint encoders.

**`--robot_pc_hand_only`** is the hand-supplement switch: it applies `HAND_LINK_PREFIXES = ("R_",)`
as a link filter, so the segment carries the fingers and the palm and drops the arm — rarely what
occludes the grasp, never what touches the tool — along with the wrist flange, which is what
`HAND_AND_FLANGE_PREFIXES` exists to add back. It does not change the point count
— `--robot_pc_points` (default 1280) still sets that — so the same budget is spent entirely on the
hand. `--robot_pc_per_link N` divides the budget per link instead, and `--no_robot` drops the
segment altogether, leaving the camera cloud alone.

Two things to know before changing any of this. The released student is trained with **`--no_robot`**
— the `norobot.zarr` in B2 — so the hand supplement is an alternative configuration here, not what
the 79.7 % checkpoint consumes. And the filter is invisible in the data: it changes which links the
points sit on but not the array's shape, so collecting with `--robot_pc_hand_only` and deploying
without it produces a correctly shaped, silently wrong observation. Deploy takes the same four flags
for that reason — but *not* the same defaults (collect: 1280 points, no per-link split; deploy: 160
and 50), so pass them explicitly on both sides rather than relying on either.

### C. Optional: the alignment stage

Screw-hole alignment is a *second, separate* teacher, warm-started from grasp end-states rather
than continuing the first. Nothing in A or B needs it — that is what `--stage1_only` means
throughout — and no number in the student ablation depends on it.

```bash
python scripts/collect_success_data.py --headless --num_envs 4096 \
    --checkpoint <ckpt> --output collected_data/success_data.pkl
python scripts/train2.py --headless --num_envs 4096 --dataset collected_data/success_data.pkl
python scripts/play_stage2.py --checkpoint <ckpt>      # play_chained.py runs both stages
```

Deploy can then hand over to it (`--stage2_rl`), or run a student distilled from it
(`--stage2_dp3_ckpt`).

### Grading notes

The pose set is committed: `data/eval_sobol_100_pos010.npz`, 100 poses per variant, 300 in total,
each replayed exactly once, identical for every checkpoint. They come from a Sobol sequence drawn
independently of any policy, rather than reused from the teacher's successes;
`tools/make_sobol_init_poses.py -n 100` draws another set. Other deploy flags worth knowing:
`--dump_gate` (log the gate against ground-truth contact), `--dump_torque`, `--policy rl` (grade
the teacher on the same poses), `--exam_seed` (sample the exam poses at random rather than in file
order), `--dump_exam` (per-pose outcomes, needed for any paired test between two runs).

> **Collect and deploy must agree.** Both build the observation from the same `perception/` code,
> but its composition is chosen by flags (`--disable_cam2`, `--no_robot`, `--force_state`, …). A
> mismatch is silent — only the point-cloud *size* is checked against the checkpoint at startup.

The tables and figures below are produced from these runs by `tools/`: `sweep_eval.py` grades every
checkpoint of a run into a CSV, `analyze_gate.py` and `plot_*.py` turn `--dump_gate` / `--dump_torque`
dumps into the plots. `tools/isaac_python.sh` joins the conda env with Isaac Sim's kit bindings, if
your shell doesn't already.

---

## Results

In simulation, on the 300-pose Sobol exam. **What "success" measures**, exactly as
`GraspDrillEnv._check_success` computes it:

- the index fingertip is within **3 cm** of the trigger point annotated for that variant,
- **and** at least 5–6 hand links (per variant) report contact above 0.1 N,
- **and** the thumb distal link is within 3 cm of its annotated target, where the variant has one.

An episode counts as a success when that conjunction holds in **at least 20 steps of a rolling
50-step window** — accumulated within the window, not 20 consecutive steps. So the metric is a
grasp configuration *from which the trigger can be pressed*: fingertip proximity to the trigger,
not a measured trigger displacement. Nothing in the task actuates the trigger.

| Policy | Observation | Success |
|---|---|---|
| Grasp teacher | privileged state (74-d) | 93.0 % |
| Alignment teacher | privileged state + plate pose | 92.0 % |
| Grasp **student** (ours) | point cloud + proprioception | **79.7 %** † |

### Torque ablation

| Configuration | Obs. | Target | Gate | Success |
|---|:--:|:--:|:--:|--:|
| Baseline — point cloud + joint positions | | | | 70.3 % |
| Torque as observation | ✓ | | | 74.3 % |
| Torque as objective only | | ✓ | | 69.7 % ‡ |
| Observation + objective | ✓ | ✓ | | 73.3 % ‡ |
| Observation + **gate** | ✓ | | ✓ | 78.3 % † |
| **Observation + gate + objective (ours)** | ✓ | ✓ | ✓ | **79.7 %** † |

<p align="center"><img src="docs/img/ablation.png" width="86%" alt="Deployed grasp success rate versus DP3 training step, one curve per ablation condition"></p>

Each cell is the best epoch of that run, out of the 6–13 epochs evaluated — `results/success_rates.csv`
carries every one, with the metric and pose set each was measured under. Two provenance caveats come
from there, and they are why the comparison is not yet clean. **†** counted with the weaker
`Grasp success (>=1 step)` criterion — the conjunction above satisfied at *any* single step — where
the unmarked student rows use the 20-in-50 criterion, so the gated rows are scored more leniently
than the baseline they are compared against. **‡** evaluated on the teacher-solved pose set instead
of the Sobol exam, which is the easier of the two. Re-running the marked rows under the unmarked
rows' protocol is the outstanding item here. (The teacher rows above predate that CSV; their logs
were not kept.)

Read with those caveats, gating is where the gain is: +9.3 points over the baseline (239/300 vs
211/300; two-proportion z-test, p = 0.008) is the only difference in the table that separates from
the baseline. The increments it decomposes into do not, individually: torque as an observation is
+4.0 (223/300 vs 211/300, p = 0.27), and adding the gate on top of it is another +4.0 (235/300 vs
223/300, p = 0.25). Both are real in direction and neither is resolvable at n = 300, single seed — so
the honest claim is about the endpoints, not about crediting the gate with the whole +9.3. The tests
are unpaired; the per-pose outcomes needed for a paired McNemar test are what `--dump_exam` writes,
and are not kept for these runs.

On drill geometries held out from training, what is *observed* is that the contact part of the
criterion still passes — the hand encloses the handle — while the trigger part fails far more often
than on the training variants. The *hypothesis* is the annotation: a held-out drill gets its trigger
offset by hand in its own object frame, and an offset that is a few centimetres off makes the 3 cm
test fail on a grasp that looks correct. That is a claim about the labels, not about the policy, and
it is untested — it needs the held-out annotations verified against the meshes, and the failures
re-scored against verified ones, before the two can be separated. Regressing the offset from the
point cloud would remove the annotation from the loop entirely.

<p align="center">
  <img src="docs/img/stage2_alignment.png" width="62%" alt="Stage 2: the grasped drill aligned against the target plate">
</p>

---

## References

- Ze et al., **3D Diffusion Policy**, RSS 2024 — the student architecture
- He et al., **FoAR: Force-Aware Reactive Policy**, RA-L 2025 ([arXiv:2411.15753](https://arxiv.org/abs/2411.15753)) — the contact-gated branch adopted here
- Lei et al., **Learning When to See and When to Feel**, [arXiv:2604.01414](https://arxiv.org/abs/2604.01414) — suppressing torque in free motion carries most of the benefit
- Zhang et al., **TA-VLA**, [arXiv:2509.07962](https://arxiv.org/abs/2509.07962) — torque as observation vs. as auxiliary target
