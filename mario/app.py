from flask import Flask, render_template
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3 import PPO
import os

app = Flask(__name__)

class MarioAgent:
    def __init__(self):
        env = gym_super_mario_bros.make('SuperMarioBros-v0')
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = GrayScaleObservation(env, keep_dim=True)
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, 4, channels_order='last')
        self.env = env
        self.model = PPO.load('./train/best_model_1000000')

    def get_action(self, state):
        return self.model.predict(state)

mario_agent = MarioAgent()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/play')
def play():
    state = mario_agent.env.reset()
    while True:
        action, _ = mario_agent.get_action(state)
        state, _, done, _ = mario_agent.env.step(action)
        if done:
            break
    return 'Game Finished'

if __name__ == '__main__':
    app.run(debug=True)
