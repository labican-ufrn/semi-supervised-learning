from base_flexconC import BaseFlexConC


class FlexConC2(BaseFlexConC):
    def select_instances_by_rules(self):
        """
        Seleção de instâncias baseada em FlexCon-C2.
        """
        selected, predictions = self.rule_3()
        if not selected:
            selected, predictions = self.rule_4()
        return selected, predictions