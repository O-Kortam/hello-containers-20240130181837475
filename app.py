from flask import Flask
from flask import request
from flask import jsonify
from clustering import Clustered_Units
from connect import MongoConnector, DBConnector
from compute_similarities import ContentBasedSim

app = Flask(__name__)

# mongo_connection = MongoConnector().connection  # establish connection
print("Start code preparation")
clustered_units = Clustered_Units()
print("End code preparation")

@app.route('/api/getSimilars', methods=['POST'])
def searchSimilarity():
    body = request.get_json()
    id = body['id']
    print("id",id)
    lang = body["lang_id"]
    try:
        recommendations = clustered_units.get_recommendations(id, lang)
        print("recommendations 1")
        recommendations = recommendations.drop_duplicates(subset=['mod_id'], keep="first", inplace=False).head(10)
        print("recommendations 2")

        return jsonify(recommendations.to_dict('records'))
    except Exception as e:
        print("exception occured",e)
        return jsonify([])



# @app.route('/updateSimilarity')
# def updateSimilarity():
#     return ContentBasedSim(connection).update_db()


# @app.route('/insertSimilarity')
# def insertSimilarity():
#     return ContentBasedSim(connection).insert_in_db()


if __name__ == "__main__":
    app.run(debug=True)