import wandb
api = wandb.Api()
run = api.run("nico-enghardt/DepthMap/21vohsb4")
run.config["learningRate"] = 0.0002
run.update()