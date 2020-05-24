from django.contrib import admin

from .models import Sentence, Question, Activity, Decision, Report

admin.site.register(Sentence)
admin.site.register(Question)
admin.site.register(Activity)
admin.site.register(Decision)
admin.site.register(Report)