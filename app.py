import io
from flask import Flask, jsonify,send_file,request, send_from_directory
from flask_cors import CORS
import subprocess
import torch
import json
import base64
import os
import scipy.io as sio
from unmixing import Config, unmixing
from database import initDatabase, insert_lidar_data, delete_lidar_data, find_lidar_data, insert_image_data, delete_image_data, find_image_data, get_names_from_table, blur_image, blur_lidar, denoise_image, sharpen_image, add_user, add_user_lidar, add_user_image_data, get_user_lidars, get_user_image_data, delete_user, get_user_info, get_all_unmixing_records_by_name, get_user_image_datas, get_user_lidar_datas

app = Flask(__name__)
CORS(app)
# 定义一个路由，当访问根URL时，会触发这个函数
@app.route('/')
def hello_world():
    return 'Hello, World!'

# 定义一个API接口，返回JSON响应
@app.route('/api/greet', methods=['GET'])
def greet():
    # 这里可以添加更多逻辑，比如处理前端发送的数据
    response = {"message": "Hello from the backend!"}
    return jsonify(response)

#上传文件【高光谱】
@app.route('/api/upload/image', methods=['POST'])
def upload_file_image():
    print("files：",request.files)
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400
    
    filename = request.form.get('filename', 'null')
    print("filename: ", filename)

    if file:
        # 将文件对象转换为BytesIO流
        file_stream = io.BytesIO(file.read())

        try:
            initDatabase()
            # 使用scipy.io.loadmat加载.mat文件
            input_data = sio.loadmat(file_stream)
            # 存储图像及相关数据
            insert_image_data(
                filename, 
                input_data['Y'], 
                input_data['label'], 
                input_data['M1'], 
                input_data['M']
            )
            print('200')
            return jsonify({'message': 'File processed successfully', 'variables': filename}), 200
        except Exception as e:
            print(str(e))
            return jsonify({'message': 'Error processing file', 'error': str(e)}), 500


# 上传文件【雷达】
@app.route('/api/upload/lidar', methods=['POST'])
def upload_file_lidar():
    print("files：", request.files)
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    filename = request.form.get('filename', 'null')
    print("filename: ", filename)

    if file:
        # 将文件对象转换为BytesIO流
        file_stream = io.BytesIO(file.read())

        try:
            initDatabase()
            # 使用scipy.io.loadmat加载.mat文件
            input_data = sio.loadmat(file_stream)
            # 存储雷达数据
            insert_lidar_data(filename, input_data['MPN'])
            print('200')
            return jsonify({'message': 'File processed successfully', 'variables': filename}), 200
        except Exception as e:
            print(str(e))
            return jsonify({'message': 'Error processing file', 'error': str(e)}), 500

# 获取数据列表
@app.route('/api/get-lidar-names', methods=['GET'])
def get_lidar_names():
    names = get_names_from_table('lidar')
    print(names)
    return jsonify(names)

@app.route('/api/get-image-data-names', methods=['GET'])
def get_image_data_names():
    names = get_names_from_table('image_data')
    print(names)
    return jsonify(names)

# 获取解混记录
@app.route('/api/get-unmixing-records/<name>', methods=['GET'])
def get_unmixing_records(name):
    records = get_all_unmixing_records_by_name(name)
    return jsonify(records)

#获取数据详细信息
@app.route('/api/getinfo/lidar', methods=['POST'])
def get_lidar_data():
    data = request.form  # 获取 JSON 数据
    name = data.get('name')
    if not name:
        return jsonify({'error': 'Missing name'}), 400
    try:
        data = find_lidar_data(name)
        # get cow and row
        data['row'] = data['MPN'].shape[0]
        data['col'] = data['MPN'].shape[1]
        data['MPN_png'] = base64.b64encode(data['MPN_png']).decode('utf-8')
        del data['MPN']
        if data:
            return jsonify(data), 200
        else:
            return jsonify({'error': 'Data not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/getinfo/image', methods=['POST'])
def get_image_data():
    print("Received request:", request.data)  # 打印原始请求数据
    try:
        data = request.form
        print("form data:", data)  # 打印解析后的JSON数据
    except Exception as e:
        print("Error parsing form:", e)
        return jsonify({'error': 'Bad Request'}), 400

    print(data)
    name = data.get("name", "")
    if not name:
        print("no name")
        return jsonify({'error': 'Missing name'}), 400
    try:
        data = find_image_data(name)
        del data['Y']
        del data['label']
        del data['M1']
        del data['M']
        data['Y_png'] = base64.b64encode(data['Y_png']).decode('utf-8')
        if data:
            return jsonify(data), 200
        else:
            return jsonify({'error': 'Data not found'}), 404
    except Exception as e:
        print(jsonify({'error': str(e)}))
        return jsonify({'error': str(e)}), 500

# 图像处理
@app.route('/api/blur/lidar', methods=['POST'])
def process_lidar():
    data = request.form
    result = blur_lidar(data['name'], data['outName'], float(data['sigma']))
    return jsonify({"result": result})

@app.route('/api/blur/image', methods=['POST'])
def process_image():
    print(request.form)
    data = request.form
    blur_image(data['name'], data['outName'], 2)
    return jsonify({"result": "success"})

@app.route('/api/denoise/image', methods=['POST'])
def process_denoise_image():
    data = request.form
    denoise_image(data['name'], data['outName'], 2)
    return jsonify({"result": "success"})

@app.route('/api/sharpen/image', methods=['POST'])
def process_sharpen_image():
    data = request.form
    sharpen_image(data['name'], data['outName'], 2)
    return jsonify({"result": "success"})

# 开始解混
@app.route('/api/unmix',methods=['POST'])
def unmix():
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"
    torch.cuda.set_device(0)

    data = request.form  # 解析JSON数据为Python字典
    # 直接通过键访问值
    lidar_name = data.get("lidar_name", "")
    image_name = data.get("image_name", "")

    print(data)

    # 从数据库中查找数据
    lidar_data = find_lidar_data(lidar_name)
    image_data = find_image_data(image_name)
    
    # 构建并返回input_data字典
    input_data_out = {**lidar_data, **image_data}
    input_data_out["username"] = "admin"

    # 初始化 config
    config = Config()
    config.num_classes = input_data_out["num"]
    config.band = input_data_out["band"]
    config.col = input_data_out["col"]
    config.row = input_data_out["row"]
    config.learning_rate_en = float(data.get("learning_rate_en", 3e-4))  # 编码器学习率
    config.learning_rate_de = float(data.get("learning_rate_de", 1e-4))  # 解码器学习率
    config.lamda = 0  # 稀疏正则化
    config.reduction = 2  # 压缩减少
    config.delta = float(data.get("delta", 0))  # delta系数
    config.gamma = float(data.get("gamma", 0.8))  # 学习率衰减
    config.epoch = int(data.get("epoch", 20))  # 训练周期

    result = json.dumps(unmixing(config,input_data_out), indent=4)

    return result

# 在前端展示图片
@app.route('/image/<path:image_name>')
def serve_image(image_name):
    image_path = f"C:/Users/Lenovo/Desktop/IEEE_TGRS_MUNet-main/image/{image_name}"
    print(image_path)
    return send_file(image_path)

# 下载pdf
@app.route('/download_report')
def download_report():
    # 指定文件所在的目录路径
    directory = 'image'
    # 指定要下载的文件名
    filename = 'report.pdf'
    # 使用 send_from_directory 来发送文件内容，提供下载
    return send_from_directory(directory, filename, as_attachment=True)

@app.route('/api/get-all-image-data', methods=['GET'])
def get_all_image_data():
    try:
        names = get_names_from_table('image_data')  # 获取所有图片的名称
        all_data = []
        for name in names:
            data = find_image_data(name)  # 获取每张图片的详细信息
            if data:
                # 只有当有详细信息时才添加
                del data['Y']
                del data['label']
                del data['M1']
                del data['M']
                data['Y_png'] = base64.b64encode(data['Y_png']).decode('utf-8')
                all_data.append({'name': name, 'data': data})
        return jsonify(all_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/get-all-lidar-data', methods=['GET'])
def get_all_lidar_data():
    try:
        names = get_names_from_table('lidar')  # 获取所有雷达数据的名称
        all_data = []
        for name in names:
            data = find_lidar_data(name)  # 获取每个雷达数据的详细信息
            if data:
                # 只有当有详细信息时才添加
                data['row'] = data['MPN'].shape[0]
                data['col'] = data['MPN'].shape[1]
                del data['MPN']
                data['MPN_png'] = base64.b64encode(data['MPN_png']).decode('utf-8')
                all_data.append({'name': name, 'data': data})
        return jsonify(all_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
#获取指定用户的图像数据
@app.route('/api/get-image-data-for-user/<username>', methods=['GET'])
def get_image_data_for_user_route(username):
    image_data = get_user_image_datas(username)
    return jsonify(image_data)

#获取指定用户的雷达数据
@app.route('/api/get-lidar-data-for-user/<username>', methods=['GET'])
def get_lidar_data_for_user_route(username):
    lidar_data = get_user_lidar_datas(username)
    return jsonify(lidar_data)
    
#获取用户信息
@app.route('/add_user', methods=['POST'])
def add_user_route():
    add_user(
        request.form['username'],
        request.form['password_hash'],
        request.form['sex'],
        request.form['age'],
        request.form['phone'],
        request.form['email'],
        request.form['permission']
    )
    return jsonify({"status": "success"}), 201

@app.route('/add_user_lidar', methods=['POST'])
def add_user_lidar_route():
    username = request.form['username']
    lidar_name = request.form['lidar_name']
    add_user_lidar(username, lidar_name)
    return jsonify({"status": "success"}), 201

@app.route('/add_user_image_data', methods=['POST'])
def add_user_image_data_route():
    username = request.form['username']
    image_data_name = request.form['image_data_name']
    add_user_image_data(username, image_data_name)
    return jsonify({"status": "success"}), 201

@app.route('/get_user_lidars/<username>', methods=['GET'])
def get_user_lidars_route(username):
    lidars = get_user_lidars(username)
    return jsonify(lidars)

@app.route('/get_user_image_data/<username>', methods=['GET'])
def get_user_image_data_route(username):
    image_data = get_user_image_data(username)
    return jsonify(image_data)

@app.route('/delete_user', methods=['POST'])
def delete_user_route():
    username = request.form['username']
    delete_user(username)
    return jsonify({"status": "success"}), 200

@app.route('/get_user_info/<username>', methods=['GET'])
def get_user_info_route(username):
    user_info = get_user_info(username)
    if user_info:
        info_dict = dict(zip(['username', 'sex', 'age', 'phone', 'email', 'permission'], user_info))
        return jsonify(info_dict)
    else:
        return jsonify({"error": "User not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)