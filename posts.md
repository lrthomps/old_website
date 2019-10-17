---
layout: default
---

<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a> ({{ post.date | date: '%b %Y' }}):<br>      
      {{ post.content | truncatewords:50 | strip_html }}
    </li>
  {% endfor %}
</ul>
