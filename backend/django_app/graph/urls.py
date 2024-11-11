from django.urls import path
from .views import ReasoningView

urlpatterns = [
    path('reasoning/', ReasoningView.as_view(), name='reasoning'),
]
