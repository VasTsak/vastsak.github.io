import type { Site, SocialObjects } from "./types";

export const SITE: Site = {
  website: "https://vtsakalos.com/",
  author: "Vasileios Tsakalos",
  desc: "Projects, Services & Posts by Vasileios Tsakalos",
  title: "Vasileios Tsakalos",
  lightAndDarkMode: true,
  ogImage: "og.png",
  postPerPage: 6,
  scheduledPostMargin: 15 * 60 * 1000, // 15 minutes
};

export const LOCALE = {
  lang: "en", // html lang code. Set this empty and default will be "en"
  langTag: ["en-EN"], // BCP 47 Language Tags. Set this empty [] to use the environment default
} as const;

export const SOCIALS: SocialObjects = [
  {
    name: "Stripe",
    href: "https://buy.stripe.com/28obMAdue2or07udQR",
    linkTitle: "Support Vasileios Tsakalos via Stripe (monthly subscription)",
    active: true,
  },
  {
    name: "Coin",
    href: "https://donate.stripe.com/9AQcQE61M6EHcUg9AA",
    linkTitle: "Support Vasileios Tsakalos via Stripe (one-time donation)",
    active: true,
  },
  {
    name: "PayPal",
    href: "https://paypal.me/gpiskas",
    linkTitle: "Support Vasileios Tsakalos via PayPal (one-time donation)",
    active: true,
  },
  {
    name: "Coffee",
    href: "https://ko-fi.com/gpiskas",
    linkTitle: "Support Vasileios Tsakalos via Ko-Fi",
    active: true,
  },
  {
    name: "LinkedIn",
    href: "https://www.linkedin.com/in/vasileiostsakalos",
    linkTitle: "Vasileios Tsakalos on LinkedIn",
    active: true,
  },
  {
    name: "Github",
    href: "https://github.com/sponsors/VasTsak",
    linkTitle: "Vasileios Tsakalos on GitHub",
    active: true,
  },
  {
    name: "Mail",
    href: "mailto:gpiskas@gmail.com",
    linkTitle: "Send an email to Vasileios Tsakalos",
    active: true,
  },
];
