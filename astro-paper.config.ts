import { defineAstroPaperConfig } from "./src/types/config";

export default defineAstroPaperConfig({
  site: {
    url: "https://pooria.co",
    title: "",
    description: "Pooria Atarzadeh's blog about cloud engineering, MLOps and system engineering",
    author: "Pooria Atarzadeh",
    profile: "https://github.com/p0o",
    ogImage: "",
    lang: "en",
    timezone: "Europe/Copenhagen",
    dir: "ltr",
  },
  posts: {
    perPage: 5,
    perIndex: 5,
    scheduledPostMargin: 15 * 60 * 1000,
  },
  features: {
    lightAndDarkMode: true,
    dynamicOgImage: true,
    showArchives: true,
    showBackButton: true,
    editPost: {
      enabled: true,
      url: "https://github.com/p0o/pooria.co/edit/main/",
    },
    search: "pagefind",
  },
  socials: [
    { name: "github", url: "https://github.com/p0o" },
    // { name: "x", url: "https://x.com/" },
    { name: "linkedin", url: "https://www.linkedin.com/in/pooriaaz/" },
  ],
  shareLinks: [
    { name: "linkedin", url: "https://www.linkedin.com/sharing/share-offsite/?url=" },
    { name: "x", url: "https://x.com/intent/post?url=" },
  ],
});