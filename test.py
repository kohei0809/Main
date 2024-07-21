import habitat

env = habitat.Env(
    config=habitat.getconfig("config/tasks/rearrange/pick.yaml")
)
observations = env.reset()

while not env.episode_over:
    observations = env.step(env.action_space.sample())
