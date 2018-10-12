import jgraph 
import json
import urllib2

data = []
req = urllib2.Request("https://raw.githubusercontent.com/plotly/datasets/master/miserables.json")
opener = urllib2.build_opener()
f = opener.open(req)
data = json.loads(f.read())
L=len(data['links'])
Edges=[(data['links'][k]['source'], data['links'][k]['target']) for k in
    range(L)]

jgraph.draw(Edges)

print(Edges)
