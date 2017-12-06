import graphviz
from sklearn.tree import export_graphviz
data = export_graphviz(model, out_file=None, feature_names=df.columns)
graph = graphviz.Source(data)
graph.render()
