# Generated by Django 3.0.5 on 2020-05-24 12:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0005_profile_isonamsubmitted'),
    ]

    operations = [
        migrations.AddField(
            model_name='profile',
            name='is_prof_done',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='profile',
            name='last_seen_prof_idx',
            field=models.IntegerField(default=1),
        ),
        migrations.AddField(
            model_name='profile',
            name='prof_score',
            field=models.IntegerField(default=0),
        ),
    ]
