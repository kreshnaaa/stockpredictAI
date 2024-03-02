from flask import Flask, jsonify
from config import *


@app.route('/custom_query', methods=['GET'])
def custom_query():
    query = "SELECT * FROM users"
    result = db.engine.execute(query)
    data = [{'id': row[0], 'username': row[1]} for row in result]
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)
