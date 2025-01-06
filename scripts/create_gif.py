import os

import hydra
from moviepy import editor as mpy
from neural_astar.planner import NeuralAstar, VanillaAstar
from neural_astar.utils.data import create_dataloader, visualize_results
from neural_astar.utils.training import load_from_ptl_checkpoint

# ESCOLHE OS PARÂMETROS
@hydra.main(config_path="config", config_name="create_gif")

def main(config):
    dataname = os.path.basename(config.dataset) # SELECIONA O DATASET ESCOLHIDO

    if config.planner == "na": # ESCOLHE O ALGORITMO
        planner = NeuralAstar(
            encoder_input=config.encoder.input,
            encoder_arch=config.encoder.arch,
            encoder_depth=config.encoder.depth,
            learn_obstacles=False,
            Tmax=1.0,
        )
        planner.load_state_dict(
            load_from_ptl_checkpoint(f"{config.modeldir}/{dataname}")
        )

    else:
        planner = VanillaAstar()

    problem_id = config.problem_id
    savedir = f"{config.resultdir}/{config.planner}"
    os.makedirs(savedir, exist_ok=True)

    # CARREGA OS DADOS DESEJADOS
    
    dataloader = create_dataloader(
        config.dataset + ".npz",
        "test",
        100,
        shuffle=False,
        num_starts=1,
    )
    map_designs, start_maps, goal_maps, opt_trajs = next(iter(dataloader))
    outputs = planner(
        map_designs, start_maps, goal_maps, store_intermediate_results=True
    )

    # CRIA OS GIFS
     # Itera sobre uma lista de problem IDs
    for problem_id in config.problem_id:
        print(f"Processing problem_id: {problem_id}")

        outputs = planner(
            map_designs[problem_id : problem_id + 1],
            start_maps[problem_id : problem_id + 1],
            goal_maps[problem_id : problem_id + 1],
            store_intermediate_results=True,
        )

        frames = [
            visualize_results(
                map_designs[problem_id : problem_id + 1], intermediate_results, scale=4
            )
            for intermediate_results in outputs.intermediate_results
        ]
        clip = mpy.ImageSequenceClip(frames + [frames[-1]] * 15, fps=30)
        clip.write_gif(f"{savedir}/video_{dataname}_{problem_id:04d}.gif")
        print(f"Saved GIF for problem_id {problem_id} at {savedir}/video_{dataname}_{problem_id:04d}.gif")

if __name__ == "__main__":
    main()