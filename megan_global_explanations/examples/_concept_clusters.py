clustering = Clustering.load(archive_path)

graph_embeddings = megan.forward_graph(graph)['graph_embeddings']
clustering.score(graph_embeddings[0], channel=0)
clustering.score(graph_embeddings[1], channel=1)

# This is more or less an independent clustering object or if we can even do it 
# that we dont need to copy the data but do in place lazy resolving of the 
# attribute accesses of the original one that would also be cool.
clustering_80 = clustering.at_linkage(80) 

# It would be nice if we could access the clusters individually like this
print(cluster_id) # ch0_cl13
cluster = clustering[cluster_id]
# cluster should be a pseudo dict like structure inspired how "DictValues" 
# is returned when doing dict.values() and then it should be possible to 
# access the cluster attributes over that access object.
# -> it'd be cool if the merged clusters would also have their own ids

# This should return a tree object which represents the cluster tree and 
# it should be possible to easily traverse this tree with various operations 
# (perhaps this is a networkx graph - if that makes sense?) where the individual 
# nodes can be represented by the cluster_ids and the linkage ratios at which 
# they are merged...
cluster_tree = clustering.get_tree()

