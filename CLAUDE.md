# Overnight Execution Rules

## CRITICAL CONSTRAINTS
- ALL code must be REAL and WORKING. Never mock, stub, or fake any functionality.
- Do NOT modify any files outside `/home/mekashirskiy/rom4ik/direction-in-latent/`
- Do NOT kill any processes you did not create. There are sglang servers and other services running.
- Commit AND push frequently with meaningful messages after each milestone. Always git push after committing.
- This is a full autonomous overnight execution. Keep working until ALL goals from the research plan are reached.
- If a dependency install fails, try alternative approaches. Do not stop.
- If training takes time, wait for it. Do not skip or mock results.
- All models, data downloads, and training runs must be real.
- Use GPU resources available on the machine. Check with `nvidia-smi` before starting.

## Project Goal
Implement the full pipeline from `directionsinlatent_research_plan.md`:
1. Direction Discovery (PCA on h-space of diffusion model)
2. Edit Generation (positive/negative image pairs)
3. Automated Labeling (CLIP/BLIP-based)
4. Evaluation and analysis
