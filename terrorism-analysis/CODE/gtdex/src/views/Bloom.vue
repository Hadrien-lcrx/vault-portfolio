<template>
  <div style="width: 100%; padding: 15px">
    <h1>Global Terrorism Database: Neo4J Bloom</h1>
    <div id="content">
      <div class="img-block" style="display: flex; flex-direction: column; align-items: flex-start">
        <img src="../assets/graph.svg" alt="Global Terrorism Database Graph Schema">
        <span class="caption"><em>Global Terrorism Database Graph Schema</em></span>
      </div>
      <div class="spacer"></div>
      <div class="spacer"></div>
      <p>
        <a href="https://neo4j.com/bloom/">Neo4j Bloom</a> is best-in-class application for
        visualizing and exploring graph databases. Visit our
        <a href="https://bloom.neo4j.io/index.html?connectURL=bolt%2Brouting%3A//0467f778.databases.neo4j.io"
           target="_blank"
        >
          GTD Graph Database Bloom
        </a> to gain new insights into the GTD.
      </p>
    </div>
    <div class="spacer"></div>
  </div>
</template>
<style lang="scss">
#content {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  width: 100%
}
.caption {
  align-self: flex-start;
  font-size: .9em;
}
</style>
