from haystack.components.evaluators import SASEvaluator

sas_evaluator = SASEvaluator()  
sas_evaluator.warm_up()

reasult = sas_evaluator.run( ground_truth_answers=["Python is a high-level general-purpose programming language that was created by Guido van Rossum"],predicted_answers=["Python is a high-level general-purpose programming language that was created by Guido van Rossum"]),

# print(reasult['individual_scores'])
# print(reasult['score'])
print(reasult)