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

# @app.route('/api/getRecs', methods=['POST'])
# def searchSimilarity():
#     body = request.get_json()
#     id = body['id']
#     print("id",id)
#     lang = body["lang_id"]
#     try:
#         print("Units clicekd by user {} : ".format(id))
#         print(clustered_units.get_user_clicks(id))
#         recommendations = clustered_units.get_user_recommended_units(id , lang)
#         print("User Recommendations !!")
#         return jsonify(recommendations.to_dict('records'))
#     except Exception as e:
#         print("exception occured",e)
#         return jsonify([])

@app.route('/api/getSimilars', methods=['POST'])
def get_similar_units():
    body = request.get_json()
    id = body['id']
    print("id",id)
    lang = body["lang_id"]
    try:
        print("Unit selected : ".format(id))
        print(clustered_units.get_unit(id))
        recommendations = clustered_units.get_similar_units(id , lang)
        return jsonify(recommendations.to_dict('records'))
    except Exception as e:
        print("exception occured",e)
        return jsonify([])

# @app.route('/api/getTrending', methods=['POST'])
# def get_trending_units():
#     try:
#         body = request.get_json()
#         lang = body["lang_id"]
#         recommendations = clustered_units.get_popular_units(lang)
#         print("Trending Items !!")
#         return jsonify(recommendations.to_dict('records'))
#     except Exception as e:
#         print("exception occured",e)
#         return jsonify([])

# @app.route('/updateSimilarity')
# def updateSimilarity():
#     return ContentBasedSim(connection).update_db()


# @app.route('/insertSimilarity')
# def insertSimilarity():
#     return ContentBasedSim(connection).insert_in_db()


if __name__ == "__main__":
    app.run(debug=True)