# Generated by Django 3.0.5 on 2020-05-27 18:53

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('game', '0017_auto_20200527_1823'),
    ]

    operations = [
        migrations.AlterField(
            model_name='activity',
            name='created_at',
            field=models.DateTimeField(default=datetime.datetime(2020, 5, 27, 18, 53, 20, 738785, tzinfo=utc)),
        ),
    ]
