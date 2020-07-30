# Generated by Django 3.0.5 on 2020-05-02 10:32

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('game', '0005_auto_20200501_2140'),
    ]

    operations = [
        migrations.AddField(
            model_name='question',
            name='mode',
            field=models.TextField(default='MODE_1'),
        ),
        migrations.AlterField(
            model_name='activity',
            name='created_at',
            field=models.DateTimeField(default=datetime.datetime(2020, 5, 2, 10, 32, 10, 670354, tzinfo=utc)),
        ),
    ]
