from env import Environment, DATA_PATH
import os
import pandas as pd
from env.util import *
from prompts.game24 import *


def get_current_numbers(y: str) -> str:
    last_line = y.strip().split('\n')[-1]
    return last_line.split('left: ')[-1].split(')')[0]


class Game24(Environment):
    def __init__(self, datadir, feedback=True, max_steps=20):
        """
                file: a csv file (fixed)
        """
        super().__init__()
        self.value_cache = {}
        path = os.path.join(DATA_PATH, '24', datadir)
        self.data = list(pd.read_csv(path)['Puzzles'])
        self.max_steps = max_steps
        self.index = 0
        self.puzzle = self.data[self.index]
        self.history = []
        self.feedbacks = []
        self.cur_step = 0
        self.feedback = feedback

    def reset(self, idx: int):
        self.index = idx
        self.puzzle = self.data[idx]
        self.history = []
        self.feedbacks = []
        self.cur_step = 0
        return {"action": "", "feedback": []}

    def check_step(self, idx, last_step, cur_step):
        print("cur_step: ", cur_step)
        try:
            if "answer" in cur_step.lower():
                print("Current step contains answer thus Checking answer")
                correct, feedback = check_answer(self.puzzle, cur_step)
                print("Answer is correct: ", correct, " Why: ", feedback)
                if not correct:
                    return f"Step {idx} tries to give an answer but it is incorrect. {feedback}", 0
                return f"Step {idx} is correct. {feedback}", 10
            else:
                # Check if the step is valid
                print("Current step does not contain answer, Checking valid move")
                correct, feedback = check_valid_move(idx, last_step, cur_step)
                print("Current step is valid: ", correct, " Why: ", feedback)
                if not correct:
                    return f"Step {idx} is illegal. {feedback}", 0

                formula = cur_step.split('left:')[0].strip("()")
                print("Checking equation, formula:", formula)
                correct, feedback = check_equation(formula)
                print("Equation is correct: ", correct, " Why: ", feedback)
                if not correct:
                    return f"Step {idx} is not correctly calculated. {feedback}", 0
                print("Checking if it is possible to reach 24")
                correct, feedback = check_twentyfour(cur_step)
                print("It is possible: ", correct, " Why: ", feedback)
                if not correct:
                    return f"Step {idx} is impossible to lead to 24. {feedback}", 0

                return f"Step {idx} is correct and can lead to 24.", 1

        except Exception as e:
            print(e)
            return f"Step {idx} is invalid.", 0

    def generate_feedback(self, action):
        print("We are now in generate_feedback\n")
        feedbacks = ["Evaluation:"]   # feedbacks for each step
        rewards = 0
        if isinstance(action, list):
            action = action[0]
        actions = action.strip(" \n").split('\n')
        idx = len(self.history)

        for action in actions:
            if idx == 0:
                last_step = self.puzzle
            else:
                last_step = self.history[-1]
            print(last_step)
            # print(action)
            if self.feedback:
                idx += 1
            feedback, reward = self.check_step(idx, last_step, action)
            if self.feedback:
                self.feedbacks.append(feedback)
                feedbacks.append(feedback)
            if reward > 0:
                if self.feedback:
                    self.history.append(action)
                rewards += reward
            else:
                break
        # if 'answer' not in steps[-1].lower():
        #     feedbacks.append("The answer is not complete.")
        total_feedback = " ".join(feedbacks) if self.feedback else None

        print("We are now done with generate feedback")
        return total_feedback, rewards

    def step(self, action):
        self.cur_step += 1
        prev_len = len(self.history)
        feedback, reward = self.generate_feedback(action)
        print("feedback in env.step: ", feedback)
        new_len = len(self.history)
        delta = new_len - prev_len + 1 if new_len < 4 else new_len - prev_len
        assert delta > 0
        done = (reward >= 10) or (self.cur_step > self.max_steps)
        answer = [f"Step {i + 1}: {x}" for i, x in enumerate(action.split('\n')[:delta]) if x != ""]
        answer = "Attempt answer: " + "\n".join(answer)
        if self.feedback:
            info = {'action': action, 'history': self.history}
            obs = {'answer': answer, 'feedback': feedback}
        else:
            info = {'action': action, 'history': []}
            obs = {'answer': answer, 'feedback': []}
        print("obs: ", obs)
        return obs, reward, done, info

    @staticmethod
    def standard_prompt_wrap(x: str, y: str = '') -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y: str = '') -> str:
        return cot_prompt.format(input=x) + y

    @staticmethod
    def propose_prompt_wrap(x: str, y: str = '') -> str:
        current_numbers = get_current_numbers(y if y else x)
        #print("current_numbers: ", current_numbers)
        if current_numbers == '24':
            #print("got in here, because one was 24")
            #print("x: ", x)
            prompt = cot_prompt.format(input=x) + 'Steps:\n' + y + "Answer: "
            # print([prompt])
        else:
            prompt = propose_prompt.format(input=current_numbers)
        return prompt

    @staticmethod
    def validation_prompt_wrap(x: str, y: str) -> str or None:
        last_line = y.strip().split('\n')[-1]
        if 'left: ' not in last_line:  # last step
            return
        if len(y.strip().split('\n')) > 1:
            prev_line = get_current_numbers(y.strip().split('\n')[-2])
        else:
            prev_line = x
        return validation_prompt.format(input=prev_line, formula=last_line)

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        last_line = y.strip().split('\n')[-1]
        if 'left: ' not in last_line:  # last step
            ans = last_line.lower().replace('answer: ', '')
            #print("gets here!!!")
            # print([value_last_step_prompt.format(input=x, answer=ans)])
            return value_last_step_prompt.format(input=x, answer=ans)
        current_numbers = get_current_numbers(y)
        return value_prompt.format(input=current_numbers)

    @staticmethod
    def validation_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        validations = [_.split('\n')[-1] for _ in value_outputs]
        if "invalid" in validations:
            return 0
        return 1

    @staticmethod
    def reflect_prompt_wrap(x: str, y: str, feedback: str) -> str:
        return reflect_prompt.format(input=x, answer=y, feedback=feedback), value_reflect_prompt.format(input=x,
                                                                                                        answer=y,
                                                                                                        feedback=feedback)

    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        #if len(y.strip().split('\n')) == 4 and 'answer' not in y.lower():
        #   return 0
        #value_names = [_.split('\n')[-1].lower() for _ in value_outputs]
        #value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}  # TODO: ad hoc
        #value = sum(value * sum(name in value_name for value_name in value_names) for name, value in value_map.items())
        #return value
        print("=== value_outputs_unwrap ===")
        print("Input x:", x)
        print("Input y:", y)
        print("Raw value_outputs:", value_outputs)

        # Step 1: Check for early exit
        stripped_lines = y.strip().split('\n')
        print("Stripped lines in y:", stripped_lines)
        
        if len(stripped_lines) == 4 and 'answer' not in y.lower():
            print("Early exit: y contains only 4 lines and no 'answer'")
        #    return 0

        # Step 2: Get last line of each value output and lowercase it
        value_names = [output.split('\n')[-1].lower() for output in value_outputs]
        print("Extracted value judgments (last lines):", value_names)

        # Step 3: Define value mapping
        value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}
        print("Value map:", value_map)

        # Step 4: Compute the final value
        total_value = 0
        for name, score in value_map.items():
            match_count = sum(name in value_name for value_name in value_names)
            print(f"Keyword '{name}' found {match_count} times. Score contribution: {match_count * score}")
            total_value += match_count * score

        print("Total value score:", total_value)
        return total_value
