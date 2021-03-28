class ENV:
    def __init__(self, grid_number, origin_position):
        self.environment_space = list(range(grid_number))
        self.state_dim = 1

        self.destination_position = grid_number

        self.origin_position = origin_position
        self.now_position = origin_position

        self.action_space = ['left', 'right']
        self.action_dim = len(self.action_space)
        self.step_counter = 0

    def step(self, action):
        if action == 'left':
            self.now_position = self.now_position - 1
            if self.now_position < 0:
                self.now_position = 0
        else:
            self.now_position = self.now_position + 1
            if self.now_position > 6:
                self.now_position = 6

        reward = -(self.destination_position - self.now_position - 1)

        if self.now_position == (self.destination_position - 1):
            reward = 10
            done = True
        else:
            done = False
            self.step_counter += 1
        info = ''
        return self.now_position, reward, done, info

    def reset(self):
        self.now_position = self.origin_position
        self.step_counter = 0
        return self.now_position

    def render(self):
        print('当前状态为**************''')
        render = ['='] * len(self.environment_space)
        render[self.now_position] = '*'
        render = ''.join(render)
        print(render)


