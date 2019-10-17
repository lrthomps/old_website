---
layout: default
---

<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a> ({{ post.date | date: '%b %Y' }}):
      
      {{ post.summary }}
    </li>
  {% endfor %}
</ul>
