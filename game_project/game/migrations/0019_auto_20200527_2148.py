# Generated by Django 3.0.5 on 2020-05-27 21:48

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('game', '0018_auto_20200527_1853'),
    ]

    operations = [
        migrations.AlterField(
            model_name='activity',
            name='created_at',
            field=models.DateTimeField(default=datetime.datetime(2020, 5, 27, 21, 48, 18, 7373, tzinfo=utc)),
        ),
    ]