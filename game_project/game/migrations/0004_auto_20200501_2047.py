# Generated by Django 3.0.5 on 2020-05-01 20:47

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('game', '0003_auto_20200501_2023'),
    ]

    operations = [
        migrations.AlterField(
            model_name='activity',
            name='created_at',
            field=models.DateTimeField(default=datetime.datetime(2020, 5, 1, 20, 47, 36, 347531, tzinfo=utc)),
        ),
    ]
