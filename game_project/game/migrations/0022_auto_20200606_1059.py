# Generated by Django 3.0.5 on 2020-06-06 10:59

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('game', '0021_auto_20200531_0903'),
    ]

    operations = [
        migrations.AlterField(
            model_name='activity',
            name='created_at',
            field=models.DateTimeField(default=datetime.datetime(2020, 6, 6, 10, 59, 27, 382902, tzinfo=utc)),
        ),
    ]