import app from './app.js';
import { connectDb } from './db.js';
import { Tenant } from './models/Tenant.js';

const PORT = process.env.PORT || 4000;

async function start() {
  await connectDb();

  if (process.env.TENANT_DEFAULT) {
    const slug = process.env.TENANT_DEFAULT.toLowerCase();
    await Tenant.findOneAndUpdate(
      { slug },
      { slug, name: slug },
      { upsert: true, new: true, setDefaultsOnInsert: true }
    );
    // eslint-disable-next-line no-console
    console.log(`Default tenant ensured: ${slug}`);
  }

  app.listen(PORT, () => {
    // eslint-disable-next-line no-console
    console.log(`Backend listening on port ${PORT}`);
  });
}

start().catch((err) => {
  // eslint-disable-next-line no-console
  console.error('Failed to start server', err);
  process.exit(1);
});

