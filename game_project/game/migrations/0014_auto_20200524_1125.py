# Generated by Django 3.0.5 on 2020-05-24 11:25

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('game', '0013_auto_20200524_1121'),
    ]

    operations = [
        migrations.CreateModel(
            name='ProficiencySentence',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('index', models.IntegerField()),
                ('text', models.TextField()),
                ('status', models.TextField()),
            ],
        ),
        migrations.AlterField(
            model_name='activity',
            name='created_at',
            field=models.DateTimeField(default=datetime.datetime(2020, 5, 24, 11, 25, 47, 757167, tzinfo=utc)),
        ),
    ]
