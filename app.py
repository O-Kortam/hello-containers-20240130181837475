from flask import Flask
from flask import request
from flask import jsonify
from connect import MongoConnector
from compute_similarities import ContentBasedSim

app = Flask(__name__)

connection = MongoConnector().connection  # establish connection


@app.route('/getRecommendation', methods=['POST'])
def searchSimilarity():
    body = request.get_json()
    id = body['id']
    try:
        return jsonify(connection.chatbotDataBase['Unit_Similarities'].find_one({"_id": id})['recommended'])
    except:
        return jsonify([])



@app.route('/updateSimilarity')
def updateSimilarity():
    return ContentBasedSim(connection).update_db()


@app.route('/insertSimilarity')
def insertSimilarity():
    return ContentBasedSim(connection).insert_in_db()


if __name__ == "__main__":
    app.run(debug=True)
