{{ objname | escape | underline}}


{% if objtype == "function" -%}

.. currentmodule:: {{ module }}

.. autofunction:: {{ objname }}

{%- elif objtype == "class" -%}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :undoc-members:
   :no-inherited-members:
   :special-members: __call__
   :member-order: bysource

{%- else -%}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}

{%- endif -%}
