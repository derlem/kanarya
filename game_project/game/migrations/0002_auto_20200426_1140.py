# Generated by Django 3.0.5 on 2020-04-26 11:40

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('game', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='activity',
            name='created_at',
            field=models.DateTimeField(default=datetime.datetime(2020, 4, 26, 11, 40, 50, 545659, tzinfo=utc)),
        ),
    ]
