import wandb
api = wandb.Api()
#run = api.run("nico-enghardt/DepthMap/21vohsb4")
#run.config["learningRate"] = 0.0002
#run.update()

artifact = api.artifact('DepthMap/Dostojewski:v5')

artifact.metadata["trainedSteps"] = 622
artifact.metadata["trainLoss"] = 1.91

artifact.save()