from django.urls import path
from deepart import views, nest

urlpatterns = [
    path('image_generate', views.image_generate, name='image_generate'),
    path('nest_generate', nest.nest_generate, name='nest_generate'),
    path('nest_exterior', nest.nest_exterior, name='nest_exterior'),
    path('nest_stylematch', nest.nest_stylematch, name='nest_stylematch'),
    path('analyze_ref', nest.analyze_ref, name='analyze_ref'),
    path('nest_retouch', nest.nest_retouch, name='nest_retouch'),
    path('suggest_brief', nest.suggest_brief, name='suggest_brief'),
    path('nest_addpool', nest.nest_addpool, name='nest_addpool'),
    path('addpool_submit', nest.addpool_submit, name='addpool_submit'),
    path('addpool_result/<str:jid>', nest.addpool_result, name='addpool_result'),
]