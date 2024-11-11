from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .model_utils import load_model,  infer_reasoning
import os
import networkx as nx

class ReasoningView(APIView):
    def post(self, request):
        

        query_text = request.data.get('query')
        if not query_text:
            return Response({"error": "No query provided"}, status=status.HTTP_400_BAD_REQUEST)

        model = load_model()
        base_dir = os.path.dirname(__file__)  # Directory where the script is located
        data_path = os.path.join(base_dir, 'data', 'graph_data.graphml')  # Adjust the path

        # Load the graph data
        graph_data = nx.read_graphml(data_path)

        
        

        print("nodes:", graph_data.nodes)
       
       

        # Perform reasoning
        results = infer_reasoning(model, query_text, graph_data)

        response = []
        for node, score in results:
            node_data = graph_data.nodes[node]
            response.append({
                "concept": node,
                "similarity_score": score,
                "definition": node_data.get('definition', 'N/A'),
                "synonym": node_data.get('synonym', 'N/A'),
                "reference": node_data.get('reference', 'N/A'),
                "process": node_data.get('process', 'N/A')
            })

        return Response(response, status=status.HTTP_200_OK)
