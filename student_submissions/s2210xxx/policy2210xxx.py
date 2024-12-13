from policy import Policy
from student_submissions.s2210xxx.policy2313621_2310807_2211173_2311601_2312401 import BestFit, Heuristic

'''
Do không truyền observation vào hàm get_action, nên trước khi hoàn thành ep, ta sẽ kiểm tra số lượng sản phẩm còn lại, 
nếu bằng 1 thì gán action = False để khởi tạo lại.
'''

class Policy2210xxx(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_1 = False
        self.policy_2 = False
        self.action = False
        if policy_id == 1:
            self.policy_1 = True
        elif policy_id == 2:
            self.policy_2 = True

    def get_action(self, observation, info):
        if self.policy_1:
            if not self.action: # Nếu action = False thì khởi tạo lại 
                self.heuristic = Heuristic(observation)
                self.action = True
            result = self.heuristic.get_action(observation, info)
            if sum(product["quantity"] for product in observation["products"]) == 1: # Nếu số lượng sản phẩm còn lại bằng 1 thì gán action = False
                self.action = False
            return result

        if self.policy_2:
            if not self.action:
                self.bestfit = BestFit(observation)
                self.action = True
            result = self.bestfit.get_action(observation, info)
            if sum(product["quantity"] for product in observation["products"]) == 1:
                self.action = False
            return result
