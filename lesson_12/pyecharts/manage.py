from pyecharts.charts import Bar, Scatter3D
from pyecharts import options as opts
from pyecharts.faker import Faker
from flask import Flask, render_template

# 创建Flask实例
app = Flask(__name__)
# 设置路由
@app.route("/bar")
def show_pyecharts():
    bar = (
        Bar()
            .add_xaxis(Faker.choose())
            .add_yaxis("销售额", Faker.values())
        )
    # print(bar.render_embed())
    # print(bar.dump_options())
    return render_template("show_pyecharts.html", bar_data=bar.dump_options())

if __name__ == "__main__":
    # 设置为Debug模式
    app.run(debug = True)
