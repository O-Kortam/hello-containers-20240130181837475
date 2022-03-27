from flask import Flask
from flask import request
from flask import jsonify
from clustering import Clustered_Units
from connect import MongoConnector, PostgresConnector
from compute_similarities import ContentBasedSim

app = Flask(__name__)

# mongo_connection = MongoConnector().connection  # establish connection
clustered_units = Clustered_Units()

@app.route('/api/getSimilars', methods=['POST'])
def searchSimilarity():
    body = request.get_json()
    id = body['id']
    lang = body["lang_id"]
    try:
        recommendations = clustered_units.get_recommendations(id, lang)
        return jsonify(recommendations.to_dict('records'))
    except:
        return jsonify([])



# @app.route('/updateSimilarity')
# def updateSimilarity():
#     return ContentBasedSim(connection).update_db()


# @app.route('/insertSimilarity')
# def insertSimilarity():
#     return ContentBasedSim(connection).insert_in_db()


if __name__ == "__main__":
    app.run(debug=True)
