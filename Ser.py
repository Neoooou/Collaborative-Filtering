from flask import Flask, jsonify
from flask import request
from CollaborativeFiltering.UserBasedCF import UserBasedCF

app = Flask(__name__)
ubcf = UserBasedCF()


@app.route('/get_recmd', methods=['POST'])
def recmd():
    if not request.is_json:
        return jsonify({'msg':'Missing JOSN in request'}), 400
    json_data = request.get_json()

    if 'user_id' not in json_data.keys():
        return jsonify({'msg':'Missing user_id in the data'}), 400
    user_id = json_data.get('user_id')

    item_nums = json_data.get('item_nums', 10)
    recmd_items = ubcf.recommend(user_id,item_nums=item_nums)

    recmd_items = [k for k in recmd_items.keys()]
    return jsonify({"recmd_items": recmd_items})

if __name__ == '__main__':
    app.run()
