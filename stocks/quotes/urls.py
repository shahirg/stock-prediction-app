from django.urls import path
from . import views

urlpatterns = [
    path('',views.home, name="home"),
    path('about',views.about, name="about"),
    path('add_stock',views.add_stock,name="add_stock"),
    path('motor_stock',views.motor_stock,name='motor_stock'),
    path('pharma_stock',views.pharma_stock,name='pharma_stock'),
    path('delete/<stock_id>/<page>', views.delete,name="delete"),
    path('stock_model',views.stock_model,name="stock_model"),
]