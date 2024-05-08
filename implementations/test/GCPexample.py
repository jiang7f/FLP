from quBLP.problemtemplate import GraphColoringProblem as GCP
gcp = GCP(2,[[0,1]], False)
gcp.optimize()
print(gcp.solution)