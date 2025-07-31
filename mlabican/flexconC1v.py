from base_flexconC import BaseFlexConC


class FlexConC1v(BaseFlexConC):
    def select_instances_by_rules(self):
        """
        Seleção de instâncias baseada em FlexCon-C1(v).
        """
        self.update_committee()
        selected, predictions = self.rule_1()
        if not selected:
            selected, predictions = self.rule_2()
        return selected, predictions