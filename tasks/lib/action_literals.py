import math

import pandas as pd
import sys

import scipy
from cleverminer import cleverminer


class action_literals:

    version="0.37"
    clm2 = None
    debug = False
    row_count=None

    def __init__(self,clm):
        self.clm2 = clm
        self.row_count=clm.data['rows_count']

    class bcolors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKCYAN = '\033[96m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        BLACK = '\033[0;30m'
        RED = '\033[0;31m'
        GREEN = '\033[0;32m'
        BROWN = '\033[0;33m'
        BLUE = '\033[0;34m'
        PURPLE = '\033[0;35m'
        CYAN = '\033[0;36m'
        GREY = '\033[0;37m'

        DARK_GREY = '\033[1;30m'
        LIGHT_RED = '\033[1;31m'
        LIGHT_GREEN = '\033[1;32m'
        YELLOW = '\033[1;33m'
        LIGHT_BLUE = '\033[1;34m'
        LIGHT_PURPLE = '\033[1;35m'
        LIGHT_CYAN = '\033[1;36m'
        WHITE = '\033[1;37m'

        RESET = "\033[0m"



    canbeignored = bcolors.RED
    mightbeignored = bcolors.YELLOW
    cannotbeignored = bcolors.GREEN


    def _get_attribute_type(self,cedent=None, attribute=None):
        if (cedent is None or attribute is None):
            print(f"ERROR IN INTERNAL CALL. Either cedent ({cedent}) or attribute {attribute} is None")
            return
        taskinfo = self.clm2.result['taskinfo']
        cedent = taskinfo[cedent]['attributes']
        res = None
        for attr in cedent:
            if attr['name']==attribute:
                return attr['type']
        print(f"ERROR: Attribute {attribute} not found.")
        return None

    def _get_rule_ff(self,rule_id=None,cedent=None,ignore_var=None,ignore_cat=None):
        #gets a fourfold table for a rule where variable or category is ignored
        if rule_id is None:
            print("ERROR:Rule id is not specified")
            return None
        actrule = self.clm2.result['rules'][rule_id-1]
        datalabels = self.clm2.result['datalabels']
        cedents = {}

        for v in actrule['trace_cedent_dataorder']:
            cedent_name=v
            cedent_list=actrule['trace_cedent_dataorder'][v]
            val_list = actrule['traces'][v]
            order = 0
            maxval = 2 ** self.row_count -1
            cedentval = maxval

            for idx in cedent_list:
                var_val_list = val_list[order]
                order += 1
                if (datalabels['varname'][idx]==ignore_var and ignore_cat is None and cedent==v):
                    if self.debug:
                        print(f"...iv... Varname: {datalabels['varname'][idx]} will be ignored")
                else:
                    varval = 0
                    cnt=0
                    for val_idx in var_val_list:
                        if self.debug:
                            if str(datalabels['catnames'][idx][val_idx]) == ignore_cat and v==cedent:
                                print(f"...i... Varname: {datalabels['varname'][idx]} Category: {datalabels['catnames'][idx][val_idx]} will be ignored.")
                            else:
                                print(f"...i... Varname: {datalabels['varname'][idx]} Category: {datalabels['catnames'][idx][val_idx]} will be APPLIED.")
                        if not(str(datalabels['catnames'][idx][val_idx]) == ignore_cat and v == cedent):
                            varval = varval | self.clm2.data['dm'][idx][val_idx]
                            cnt+=1
                    if cnt>0:
                        cedentval = cedentval & varval

            cedents[cedent_name]=cedentval


        ff = []
        ff.insert(0,(cedents['ante']&cedents['succ']&cedents['cond']).bit_count())
        ff.insert(1,(cedents['ante']&(maxval-cedents['succ'])&cedents['cond']).bit_count())
        ff.insert(2,((maxval-cedents['ante'])&cedents['succ']&cedents['cond']).bit_count())
        ff.insert(3,((maxval-cedents['ante'])&(maxval-cedents['succ'])&cedents['cond']).bit_count())
        return ff

    def calc_z_score(self,ff1,ff2):
        # calculate CUSTOM METRIC for rules: z-score for two sample test of equality of the mean in two binomial distributions
        p1 = None
        if (ff1[0] + ff1[1])>0:
            p1 = ff1[0] / (ff1[0] + ff1[1])
        else:
            p1 = 0
        p2 = None
        if ff2[0] + ff2[1] > 0:
            p2 = ff2[0] / (ff2[0] + ff2[1])
        else:
            p2 = 0
        n1 = ff1[0] + ff1[1]
        n2 = ff2[0] + ff2[1]

        p = (n1 * p1 + n2 * p2) / (n1 + n2)
        z = None
        if n1>0 and n2>0:
            if p * (1 - p) * (1 / n1 + 1 / n2) >0:
                z = (p1 - p2) / math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
            else:
                z = 0
        else:
            z=0
        p_val = scipy.stats.norm.cdf(z)
        pimratioabs = 0
        if p2>0:
            pimratioabs = p1/p2
        if pimratioabs<1:
            pimratioabs=p2/p1
        return z ,p_val, pimratioabs

    def _get_literal_importance(self,rule_id=None,print_rule=True,print_details=True):
        if rule_id is None:
            print("ERROR:Rule id is not specified")
            return None

        if print_details:
            print(f'\nDetailed analysis of rule {rule_id}: {self.clm2.get_ruletext(rule_id)}')
        ff_full = ff=self._get_rule_ff(rule_id=rule_id)

        ff_texts={}

        actrule = self.clm2.result['rules'][rule_id-1]
        datalabels = self.clm2.result['datalabels']
        for v in actrule['trace_cedent_dataorder']:
            ff_text = ''
            cedent_name=v
            cedent_list=actrule['trace_cedent_dataorder'][v]
            val_list = actrule['traces'][v]
            order = 0
            maxval = 2 ** self.row_count -1
            cedentval = maxval

            for idx in cedent_list:
                if order>0:
                    ff_text = ff_text + ' & '
                vn = datalabels['varname'][idx]
                tp = self._get_attribute_type(v,vn)
                ff = self._get_rule_ff(rule_id,v,vn)
                conf_full = ff_full[0] / (ff_full[0] + ff_full[1])
                conf = ff[0] / (ff[0] + ff[1])
                base_full = ff_full[0]
                base = ff[0]
                z_score,p_val,pimratioabs = self.calc_z_score(ff_full,ff)
                if abs(z_score)>2*1.96:#20:#pimratioabs>1.5:
                    if print_details:
                        print(f"   When Var {self.cannotbeignored}{vn}{self.bcolors.ENDC} is ignored: {self.cannotbeignored}confidence {conf_full:.3f} -> {conf:.3f}, base {base_full} -> {base}{self.bcolors.ENDC}, pval {p_val:.5f}, z_score {z_score:.3f}, pimratioabs {pimratioabs:.3f}")
                    ff_text= ff_text + self.cannotbeignored+vn+self.bcolors.ENDC
                elif abs(z_score)>1.96:#pimratioabs>1.2:
                    if print_details:
                        print(f"   When Var {self.mightbeignored}{vn}{self.bcolors.ENDC} is ignored: {self.mightbeignored}confidence {conf_full:.3f} -> {conf:.3f}, base {base_full} -> {base}{self.bcolors.ENDC}, pval {p_val:.5f}, z_score {z_score:.3f}, pimratioabs {pimratioabs:.3f}")
                    ff_text= ff_text + self.mightbeignored+vn+self.bcolors.ENDC
                else:
                    if print_details:
                        print(f"   When Var {self.canbeignored}{vn}{self.bcolors.ENDC} is ignored: confidence {self.canbeignored}{conf_full:.3f} -> {conf:.3f}, base {base_full} -> {base}{self.bcolors.ENDC}, pval {p_val:.5f}, z_score {z_score:.3f}, pimratioabs {pimratioabs:.3f}")
                    ff_text= ff_text+self.canbeignored+ vn+self.bcolors.ENDC
                ff_text=ff_text +'('


                var_val_list=val_list[order]
                order2=0

                #determine eligible indices based on variable type

                eligible_val_list = None

                if tp=='subset':
                    eligible_val_list = var_val_list
                elif tp=='rcut' or tp == 'lcut':
                    eligible_val_list = var_val_list[-1:]
                elif tp=='seq':
                    eligible_val_list = [var_val_list[0],var_val_list[-1]]
                elif tp =='one':
                    eligible_val_list=var_val_list
                else:
                    print(f"ERROR: Unsupported variable type: {tp}. Will treat this type as a subset.")
                    eligible_val_list = var_val_list

                if len(var_val_list)==1:
                    eligible_val_list = []

                for val_idx in var_val_list:

                    if order2 > 0:
                        ff_text = ff_text + ' '
                    cn = datalabels['catnames'][idx][val_idx]
                    cn=str(cn)
                    ff2 = self._get_rule_ff(rule_id,v,vn,cn)
                    conf2 = None
                    if ff2[0]+ff2[1] >0:
                        conf2 = ff2[0]/(ff2[0]+ff2[1])
                    else:
                        conf2 = 0
                    base2=ff2[0]
                    z_score, p_val,pimratioabs = self.calc_z_score(ff_full, ff2)
                    if val_idx not in eligible_val_list:
                        if print_details:
                            print(f"   ... varname's {vn} category {cn} cannot be ignored due to either single category or the ordinal data type and not eligible (middle) category cannot be thrown out")
                        ff_text = ff_text + cn
                    elif abs(z_score)>20:# pimratioabs>1.5:
                        if print_details:
                            print(f"   ... when varname's {self.cannotbeignored}{vn}{self.bcolors.ENDC} category {self.cannotbeignored}{cn}{self.bcolors.ENDC} ignored : confidence {self.cannotbeignored}{conf_full:.3f} -> {conf2:.3f}, base {base_full} -> {base2}{self.bcolors.ENDC}, pval {p_val:.5f}, z_score {z_score:.3f}, pimratioabs {pimratioabs:.3f}")
                        ff_text = ff_text + self.cannotbeignored + cn + self.bcolors.ENDC
                    elif abs(z_score)>1.96: #pimratioabs>1.2:
                        if print_details:
                            print(f"   ... when varname's {self.mightbeignored}{vn}{self.bcolors.ENDC} category {self.mightbeignored}{cn}{self.bcolors.ENDC} ignored : confidence {self.mightbeignored}{conf_full:.3f} -> {conf2:.3f}, base {base_full} -> {base2}{self.bcolors.ENDC}, pval {p_val:.5f}, z_score {z_score:.3f}, pimratioabs {pimratioabs:.3f}")
                        ff_text = ff_text + self.mightbeignored + cn + self.bcolors.ENDC
                    else:
                        if print_details:
                            print(f"   ... when varname's {self.canbeignored}{vn}{self.bcolors.ENDC} category {self.canbeignored}{cn}{self.bcolors.ENDC} ignored : confidence {self.canbeignored}{conf_full:.3f} -> {conf2:.3f}, base {base_full} -> {base2}{self.bcolors.ENDC}, pval {p_val:.5f}, z_score {z_score:.3f}, pimratioabs {pimratioabs:.3f}")
                        ff_text = ff_text +self.canbeignored + cn + self.bcolors.ENDC


                    order2+=1
                order+=1
                ff_text = ff_text + ')'
            ff_texts[cedent_name]=ff_text


        final_text = ff_texts['ante'] + ' => ' + ff_texts['succ']+' | '+ff_texts['cond']
        if print_rule:
            print(f'Colored rule is {final_text}')
        return final_text

    def print_rule_literal_importance(self,rule_id=None):
        self._get_literal_importance(rule_id=rule_id,print_rule=True,print_details=True)


    def print_rulelist(self):
        print(
            f"\nLEGEND: \n  {self.cannotbeignored}ATTRIBUTE/VALUE IS VERY IMPORTANT FOR INTERPRETATION AND CANNOT BE IGNORED{self.bcolors.ENDC}\n  {self.mightbeignored}ATTRIBUTE/VALUE IS SIGNIFICANT FOR INTERPRETATION{self.bcolors.ENDC}\n  {self.canbeignored}ATTRIBUTE/VALUE SHOULD NOT BE USED IN INTERPRETATION AND CAN BE IGNORED{self.bcolors.ENDC}\n  VALUE IS NOT ELIGIBLE FOR ASSESSING THE IMPORTANCE (E.G. A MIDDLE CATEGORY OF AN ORDINAL SEQUENCE)")

        print("\n RULE_ID BASE  CONF   AAD   RULE_TEXT")
        for iii in range(self.clm2.get_rulecount()):
            ruleid = iii + 1

            ff = self.clm2.get_fourfold(ruleid)
            rule_text = self._get_literal_importance(ruleid,False,False)
            base = ff[0]
            conf = 0
            if ff[0] + ff[1] > 0:
                conf = base / (ff[0] + ff[1])
                aad = 0
                if ff[0] > 0:
                    aad = conf / ( (ff[0] + ff[2]) / sum(ff)) -1
            print(f'{ruleid:>8} {base:>5} {conf:.3f} {aad:+.3f} {rule_text}')


