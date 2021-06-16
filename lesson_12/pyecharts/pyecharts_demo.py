#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyecharts.charts import Bar
from pyecharts.faker import Faker

bar = Bar()
bar.add_xaxis(Faker.choose())
bar.add_yaxis("销售额", Faker.values())
# 生成本地html
bar.render('temp.html')
bar.render_notebook()


# In[2]:


# 直方图
from pyecharts import options as opts
from pyecharts.charts import Bar
bar = Bar().add_xaxis(Faker.cars)           .add_yaxis("京东",Faker.values())           .add_yaxis("天猫",Faker.values())           .set_global_opts(
                title_opts=opts.TitleOpts(title="电商汽车销量", pos_left="5%"),
                legend_opts=opts.LegendOpts(pos_left="30%")
            )
# bar.render("temp.html")
bar.render_notebook()
#bar.render(path='snapshot.png', pixel_ratio=3)


# In[3]:


from pyecharts.components import Table
table = Table()

headers = ["城市名", "GDP", "常驻人口", "人均GDP"]
rows = [
    ['北京', 36103, 2153.6, 167640],
    ['上海', 38701, 2428.14, 159385],
    ['江苏', 102719, 8070, 127285],
    ['福建', 43904, 3973, 110506],
    ['浙江', 64613, 5850, 110450],
    ['广东', 110761, 11521, 96138],
    ['天津', 14084, 1561.83, 90176],
]
table.add(headers, rows).set_global_opts(
    title_opts=opts.ComponentTitleOpts(title="不同城市GDP统计")
)
table.render_notebook()


# In[4]:


from pyecharts.charts import Line
line1=(
    Line() # 生成line类型图表
    .add_xaxis(Faker.choose())  # 添加x轴
    .add_yaxis('数据1',Faker.values())  # 添加y轴
    .add_yaxis('数据2',Faker.values())
    .set_global_opts(
        title_opts=opts.TitleOpts(title='PyEcharts折线图', pos_right="5%"),
        legend_opts=opts.LegendOpts(pos_right="30%")
    )
    
)
line1.render_notebook()


# In[5]:


from pyecharts.charts import Page, Grid
#grid = Grid()
grid = Grid(init_opts=opts.InitOpts(width="900px", height="400px"))

grid.add(bar, grid_opts=opts.GridOpts(pos_left="55%"))
grid.add(line1, grid_opts=opts.GridOpts(pos_right="55%"))
grid.render_notebook()


# In[6]:


# 导入输出图片工具
from pyecharts.render import make_snapshot
# 使用snapshot-selenium 渲染图片
from snapshot_selenium import snapshot
# 需要将chromedriver放到PATH中
# 输出保存为图片
make_snapshot(snapshot, grid.render(), "grid.png")


# ## 2D散点图

# In[18]:


from pyecharts.charts import EffectScatter
#es.render(path = '散点图.jpeg')
es = (
    EffectScatter()
    .add_xaxis(Faker.choose())
    .add_yaxis("", Faker.values())
    .set_global_opts(title_opts=opts.TitleOpts(title="散点图"))
)
es.render_notebook()
#es.render(path = '散点图.jpg')


# ## 3D散点图

# In[7]:


import random
from pyecharts import options as opts
from pyecharts.charts import Scatter3D
from pyecharts.faker import Faker

# 生成模拟数据
data = [(random.randint(0,50),random.randint(0,50),random.randint(0,50)) for i in range(50)]
# 设置3D散点图
scatter = (
    Scatter3D(init_opts = opts.InitOpts(width='900px',height='600px'))  #初始化
    .add("", data,
         grid3d_opts=opts.Grid3DOpts(
            width=100, depth=100, rotate_speed=10, is_rotate=True
        ))
    
    #设置全局配置项
    .set_global_opts(
        title_opts=opts.TitleOpts(title="3D散点图"),  #添加标题
        visualmap_opts=opts.VisualMapOpts(
            max_=50, #最大值
            pos_top=50, # visualMap 组件离容器上侧的距离
            range_color=Faker.visual_color  #颜色映射                                         
        )
    )
)
scatter.render("3D散点图.html")


# In[6]:


# 使用配置项需要导入相应模块
from pyecharts import options as opts
from pyecharts.charts import Map
from pyecharts.faker import Faker

city_names = ['杭州市', '宁波市', '温州市', '绍兴市', '嘉兴市', '台州市', '金华市', '湖州市', '衢州市', '丽水市', '舟山市' ]
city_gdp = [15373, 11985, 6606, 5780, 5370, 5134, 4559, 3122, 1573, 1476, 1371]

map1 = (
    Map()
    .add("浙江省各城市GDP", [list(z) for z in zip(city_names, city_gdp)], "浙江", is_map_symbol_show=True,)
    # 系列配置（标签配置）
    .set_series_opts(label_opts=opts.LabelOpts(
        # is_show=True 是否显示标签
        is_show=True,
        position='bottom',
        font_size=10,
        color= '#CC6633', #文字颜色
        font_style = 'italic' , #斜体
        font_weight = None,
        font_family = None,
        rotate = '15', #倾斜角度 [-90, 90]
        margin = 20, # 刻度标签与轴线之间的距离
        interval = None,
        horizontal_align = 'center',
        vertical_align = None,
        ))

    .set_global_opts(
        title_opts=opts.TitleOpts(title=""),
        visualmap_opts=opts.VisualMapOpts(min_=1000, max_=10000,range_color=['#d1d1d1','#dcdc22'])
    )
)
map1.render_notebook()


# In[8]:


from pyecharts.charts import Tab
# 创建组合类对象
tab = Tab()
# 在组合对象中添加需要组合的图表对象
tab.add(line1, "折线图")
tab.add(bar, "直方图")
tab.add(map1, "地图")
tab.add(table, "图表")
# 5. 渲染数据
tab.render_notebook()


# In[22]:


Faker.choose()


# In[24]:


Faker.values()


# In[25]:


Faker.cars


# In[27]:


Faker.days_attrs


# In[ ]:


Faker.cars # 随机各种中文汽车品牌的列表
Faker.visual_color # 随机颜色列表
Faker.days_attrs #  'number天'字符串列表
Faker.clock # 时间字符串列表
Faker.dogs # 随机各种狗的列表
Faker.guangdong_city # 广东省下面7个市的固定列表
Faker.img_path(r'C:\abc.jpg') # 返回图片路径
Faker.week_en # 英文的星期一到日

