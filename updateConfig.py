import wandb
api = wandb.Api()
run = api.run("nico-enghardt/PointRecognition/3oga2x3t")
run.config["batch-size"] = 2281
run.update()