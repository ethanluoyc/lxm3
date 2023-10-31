import unittest

import gym


class MujocoPyTest(unittest.TestCase):
    def test_run_env(self):
        env = gym.make("HalfCheetah-v2")
        env.reset()
        env.step(env.action_space.sample())
        env.render("rgb_array")


if __name__ == "__main__":
    unittest.main()
